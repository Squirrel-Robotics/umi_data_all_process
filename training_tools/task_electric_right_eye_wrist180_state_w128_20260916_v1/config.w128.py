"""See _CONFIGS for the list of available configs."""

import abc
from collections.abc import Sequence
import dataclasses
import difflib
import logging
import pathlib
from typing import Any, Literal, Protocol, TypeAlias

import etils.epath as epath
import flax.nnx as nnx
from typing_extensions import override
import tyro

import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.models.pi0_fast as pi0_fast
import openpi.models.tokenizer as _tokenizer
import openpi.policies.aloha_policy as aloha_policy
import openpi.policies.cx002_policy as cx002_policy
import openpi.policies.droid_policy as droid_policy
import openpi.policies.libero_policy as libero_policy
import openpi.shared.download as _download
import openpi.shared.normalize as _normalize
import openpi.training.droid_rlds_dataset as droid_rlds_dataset
import openpi.training.misc.polaris_config as polaris_config
import openpi.training.misc.roboarena_config as roboarena_config
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms

ModelType: TypeAlias = _model.ModelType
_CX002_TASK1_PROMPT = "Nest the three colored cups together."
_CX002_TASK2_PROMPT = "Nest the three colored cups together."
_CX002_TASK3_PROMPT = "Place the right block in the middle box, then stack the left block on top of it."
_CX002_TASK4_PROMPT = "Put the small squares from the table into the container."
_CX002_TASK5_PROMPT = "Turn the package over."
_UMI_LATEST100_PROMPT = "placeholder_single_task"
_UMI_BOX_IN_OUT_PROMPT = "Put the object into the box, then take it out."
_UMI_TASK_V1_PROMPT = "Put the object into the box."
_UMI_TASK_V1_NEW_PROMPT = "Put the object on the box, then return it back."
_UMI_TASK_V3_PROMPT = "Put the object on the box, then take it down."

# Work around a tyro issue with using nnx.filterlib.Filter directly.
Filter: TypeAlias = nnx.filterlib.Filter


@dataclasses.dataclass(frozen=True)
class AssetsConfig:
    """Determines the location of assets (e.g., norm stats) that will be used to set up the data pipeline.

    These assets will be replicated inside the checkpoint under the `assets/asset_id` directory.

    This can be used to load assets from a different checkpoint (e.g., base model checkpoint) or some other
    centralized location. For example, to load the norm stats for the Trossen robot from the base model checkpoint
    during fine-tuning, use:

    ```
    AssetsConfig(
        assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
        asset_id="trossen",
    )
    ```
    """

    # Assets directory. If not provided, the config assets_dirs will be used. This is useful to load assets from
    # a different checkpoint (e.g., base model checkpoint) or some other centralized location.
    assets_dir: str | None = None

    # Asset id. If not provided, the repo id will be used. This allows users to reference assets that describe
    # different robot platforms.
    asset_id: str | None = None


@dataclasses.dataclass(frozen=True)
class DataConfig:
    # LeRobot repo id. If None, fake data will be created.
    repo_id: str | None = None
    # Directory within the assets directory containing the data assets.
    asset_id: str | None = None
    # Contains precomputed normalization stats. If None, normalization will not be performed.
    norm_stats: dict[str, _transforms.NormStats] | None = None

    # Used to adopt the inputs from a dataset specific format to a common format
    # which is expected by the data transforms.
    repack_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Data transforms, typically include robot specific transformations. Will be applied
    # before the data is normalized. See `model.Observation` and `model.Actions` to learn about the
    # normalized data.
    data_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Model specific transforms. Will be applied after the data is normalized.
    model_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantile_norm: bool = False

    # Names of keys that will be used by the data loader to generate the action sequence. The length of the
    # sequence is defined by the `action_horizon` field in the model config. This should be adjusted if your
    # LeRobot dataset is using different keys to represent the action.
    action_sequence_keys: Sequence[str] = ("actions",)

    # If true, will use the LeRobot dataset task to define the prompt.
    prompt_from_task: bool = False

    # Optional boolean/scalar LeRobot feature used to select trainable anchor
    # rows before shuffling or batching.  The underlying dataset remains
    # intact, so delta_timestamps can still load future action targets.
    sample_filter_key: str | None = None

    # Optional transformed batch key containing a boolean [B,H] action-slot
    # padding mask. When configured, the loader fails closed if the mask is
    # missing or does not match the action batch/horizon axes.
    action_padding_mask_key: str | None = None

    # Only used for RLDS data loader (ie currently only used for DROID).
    rlds_data_dir: str | None = None
    # Action space for DROID dataset.
    action_space: droid_rlds_dataset.DroidActionSpace | None = None
    # List of datasets to sample from: name, version, weight, and optionally filter_dict_path
    datasets: Sequence[droid_rlds_dataset.RLDSDataset] = ()


class GroupFactory(Protocol):
    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        """Create a group."""


@dataclasses.dataclass(frozen=True)
class ModelTransformFactory(GroupFactory):
    """Creates model transforms for standard pi0 models."""

    # If provided, will determine the default prompt that be used by the model.
    default_prompt: str | None = None

    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        match model_config.model_type:
            case _model.ModelType.PI0:
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI05:
                assert isinstance(model_config, pi0_config.Pi0Config)
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                            discrete_state_input=model_config.discrete_state_input,
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI0_FAST:
                tokenizer_cls = (
                    _tokenizer.FASTTokenizer
                    if model_config.fast_model_tokenizer is None
                    else model_config.fast_model_tokenizer
                )
                tokenizer_kwargs = (
                    {} if model_config.fast_model_tokenizer_kwargs is None else model_config.fast_model_tokenizer_kwargs
                )
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizeFASTInputs(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                        ),
                    ],
                    outputs=[
                        _transforms.ExtractFASTActions(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                            action_horizon=model_config.action_horizon,
                            action_dim=model_config.action_dim,
                        )
                    ],
                )


@dataclasses.dataclass(frozen=True)
class DataConfigFactory(abc.ABC):
    # The LeRobot repo id.
    repo_id: str = tyro.MISSING
    # Determines how the assets will be loaded.
    assets: AssetsConfig = dataclasses.field(default_factory=AssetsConfig)
    # Base config that will be updated by the factory.
    base_config: tyro.conf.Suppress[DataConfig | None] = None

    @abc.abstractmethod
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        """Create a data config."""

    def create_base_config(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repo_id = self.repo_id if self.repo_id is not tyro.MISSING else None
        asset_id = self.assets.asset_id or repo_id
        return dataclasses.replace(
            self.base_config or DataConfig(),
            repo_id=repo_id,
            asset_id=asset_id,
            norm_stats=self._load_norm_stats(epath.Path(self.assets.assets_dir or assets_dirs), asset_id),
            use_quantile_norm=model_config.model_type != ModelType.PI0,
        )

    def _load_norm_stats(self, assets_dir: epath.Path, asset_id: str | None) -> dict[str, _transforms.NormStats] | None:
        if asset_id is None:
            return None
        try:
            data_assets_dir = str(assets_dir / asset_id)
            norm_stats = _normalize.load(_download.maybe_download(data_assets_dir))
            logging.info(f"Loaded norm stats from {data_assets_dir}")
            return norm_stats
        except FileNotFoundError:
            logging.info(f"Norm stats not found in {data_assets_dir}, skipping.")
        return None


@dataclasses.dataclass(frozen=True)
class FakeDataConfig(DataConfigFactory):
    repo_id: str = "fake"

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return DataConfig(repo_id=self.repo_id)


@dataclasses.dataclass(frozen=True)
class SimpleDataConfig(DataConfigFactory):
    # Factory for the data transforms.
    data_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=GroupFactory)
    # Factory for the model transforms.
    model_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=ModelTransformFactory)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            data_transforms=self.data_transforms(model_config),
            model_transforms=self.model_transforms(model_config),
        )


@dataclasses.dataclass(frozen=True)
class LeRobotAlohaDataConfig(DataConfigFactory):
    # If true, will convert joint dimensions to deltas with respect to the current state before passing to the model.
    # Gripper dimensions will remain in absolute values.
    use_delta_joint_actions: bool = True
    # If provided, will be injected into the input data if the "prompt" key is not present.
    default_prompt: str | None = None
    # If true, this will convert the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model. People who
    # use standard Aloha data should set this to true.
    adapt_to_pi: bool = True

    # Repack transforms.
    repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(
        default=_transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": {"cam_high": "observation.images.top"},
                        "state": "observation.state",
                        "actions": "action",
                    }
                )
            ]
        )
    )
    # Action keys that will be used to read the action sequence from the dataset.
    action_sequence_keys: Sequence[str] = ("action",)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        data_transforms = _transforms.Group(
            inputs=[aloha_policy.AlohaInputs(adapt_to_pi=self.adapt_to_pi)],
            outputs=[aloha_policy.AlohaOutputs(adapt_to_pi=self.adapt_to_pi)],
        )
        if self.use_delta_joint_actions:
            delta_action_mask = _transforms.make_bool_mask(6, -1, 6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=self.repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotCX002EefDataConfig(DataConfigFactory):
    """CX002 dual-EEF config based on Squirrel Robotics' relative EEF pipeline.

    Raw state/actions contain two absolute 9D EEF blocks:
    ``left xyz+rot6d, right xyz+rot6d, left gripper, right gripper`` (20D).

    ``eef_dim=6`` follows Squirrel's actual relative-Euler training configs:
    both EEF blocks are converted to xyz+rpy before the model, producing 14D
    state/actions including the two absolute grippers. ``eef_dim=9`` preserves
    xyz+rot6d and produces 20D state/actions. In either mode, EEF actions are
    relative to the current state while gripper targets stay absolute.
    """

    default_prompt: str | None = None
    eef_dim: Literal[6, 9] = 6
    use_delta_eef_actions: bool = True
    use_quantile_norm: bool | None = None

    repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(
        default=_transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": {
                            "cam_high": "observation.images.base_0_rgb",
                            "cam_left_wrist": "observation.images.left_wrist_0_rgb",
                            "cam_right_wrist": "observation.images.right_wrist_0_rgb",
                        },
                        "state": "observation.state",
                        "actions": "action",
                    }
                )
            ]
        )
    )
    action_sequence_keys: Sequence[str] = ("action",)

    @property
    def policy_action_dim(self) -> int:
        return 2 * self.eef_dim + 2

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        data_transforms = _transforms.Group(
            inputs=[cx002_policy.CX002Inputs()],
            outputs=[cx002_policy.CX002Outputs(action_dim=self.policy_action_dim)],
        )

        if self.eef_dim == 6:
            data_transforms = data_transforms.push(inputs=[_transforms.EefRot6dToEuler()])
            if self.use_delta_eef_actions:
                data_transforms = data_transforms.push(
                    inputs=[_transforms.DeltaEefEulerActions(action_dim=self.policy_action_dim)],
                    outputs=[_transforms.AbsoluteEefEulerActions(action_dim=self.policy_action_dim)],
                )
        elif self.eef_dim == 9:
            if self.use_delta_eef_actions:
                data_transforms = data_transforms.push(
                    inputs=[_transforms.DeltaEefActions(action_dim=self.policy_action_dim)],
                    outputs=[_transforms.AbsoluteEefActions(action_dim=self.policy_action_dim)],
                )
        else:
            raise ValueError(f"eef_dim must be 6 or 9, got {self.eef_dim}")

        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)
        base_config = self.create_base_config(assets_dirs, model_config)
        if self.use_quantile_norm is not None:
            base_config = dataclasses.replace(base_config, use_quantile_norm=self.use_quantile_norm)

        return dataclasses.replace(
            base_config,
            repack_transforms=self.repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotCX002RawEef6DataConfig(DataConfigFactory):
    """CX002 raw14 dual-EEF config with absolute xyz+rpy future targets.

    State and action layout is
    ``left xyz+rpy, right xyz+rpy, left gripper, right gripper`` (14D).
    Dataset action at frame ``t`` is the absolute target at ``t+1``.
    During training, every future target in an action chunk is converted to a
    base-frame translation delta and body-frame relative rotation anchored at
    the current state. Gripper targets remain absolute.
    """

    default_prompt: str | None = None
    use_delta_eef_actions: bool = True
    output_absolute_eef_actions: bool = True
    use_quantile_norm: bool | None = None

    repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(
        default=_transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": {
                            "cam_high": "observation.images.base_0_rgb",
                            "cam_left_wrist": "observation.images.left_wrist_0_rgb",
                            "cam_right_wrist": "observation.images.right_wrist_0_rgb",
                        },
                        "state": "observation.state",
                        "actions": "action",
                        "prompt": "prompt",
                    }
                )
            ]
        )
    )
    action_sequence_keys: Sequence[str] = ("action",)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        data_transforms = _transforms.Group(
            inputs=[cx002_policy.CX002Inputs()],
            outputs=[cx002_policy.CX002Outputs(action_dim=14)],
        )
        if self.use_delta_eef_actions:
            output_transforms = (
                [_transforms.AbsoluteEefEulerActions(action_dim=14)]
                if self.output_absolute_eef_actions
                else []
            )
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaEefEulerActions(action_dim=14)],
                outputs=output_transforms,
            )

        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)
        base_config = self.create_base_config(assets_dirs, model_config)
        if self.use_quantile_norm is not None:
            base_config = dataclasses.replace(base_config, use_quantile_norm=self.use_quantile_norm)
        return dataclasses.replace(
            base_config,
            repack_transforms=self.repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotCX002PdpDataConfig(DataConfigFactory):
    """CX002 PDP data whose 6D EEF actions are already relative.

    The dataset state is 20D with interleaved grippers:
    ``left xyz+rot6d, left gripper, right xyz+rot6d, right gripper``.
    The action is 14D:
    ``left delta_xyz+delta_rotvec, left gripper, right delta_xyz+delta_rotvec,
    right gripper``.

    PDP actions are per-step deltas, so applying Squirrel's absolute-to-delta
    transform again would be incorrect. State and actions are passed through
    unchanged and padded to the model's 32D action dimension later.
    """

    default_prompt: str | None = None
    repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(
        default=_transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": {
                            "cam_high": "observation.images.base_0_rgb",
                            "cam_left_wrist": "observation.images.left_wrist_0_rgb",
                            "cam_right_wrist": "observation.images.right_wrist_0_rgb",
                        },
                        "state": "observation.state",
                        "actions": "action",
                    }
                )
            ]
        )
    )
    action_sequence_keys: Sequence[str] = ("action",)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        data_transforms = _transforms.Group(
            inputs=[cx002_policy.CX002Inputs()],
            outputs=[cx002_policy.CX002Outputs(action_dim=14)],
        )
        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=self.repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotUmiSpatialEefHand24DataConfig(DataConfigFactory):
    """Latest-100 UMI data with spatial EEF12 and absolute hand12.

    The LeRobot rows store ``left EEF6, right EEF6, left hand6, right hand6``.
    Controller poses are the robot-aligned EEF proxies; hand values are
    absolute normalized Revo2 actuator positions. ``action[t]`` is the next
    absolute 24D state, so the H30 loader returns states ``t+1:t+31``. A custom
    transform makes only the first 12 EEF dimensions relative to the one
    current-state anchor using the robot-base/spatial convention:

    ``dp = p_target - p_anchor`` and
    ``dR = R_target @ R_anchor.T``.

    The final 12 hand targets remain absolute. This is intentionally different
    from ``DeltaEefEulerActions`` (body-frame rotation). The spatial transform
    below is the one and only EEF action conversion.
    """
    use_head_camera: bool = True
    default_prompt: str | None = None
    use_delta_eef_actions: bool = False
    output_absolute_eef_actions: bool = False
    use_quantile_norm: bool | None = None
    sample_filter_key: str = "sample_valid_h30"

    repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(
        default=_transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": {
                            "cam_high": "observation.images.base_0_rgb",
                            "cam_left_wrist": "observation.images.left_wrist_0_rgb",
                            "cam_right_wrist": "observation.images.right_wrist_0_rgb",
                        },
                        "state": "observation.state",
                        "actions": "action",
                        "prompt": "prompt",
                    }
                )
            ]
        )
    )
    action_sequence_keys: Sequence[str] = ("action",)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        if model_config.action_horizon != 30:
            raise ValueError(
                f"latest-100 UMI contract requires action_horizon=30, got {model_config.action_horizon}"
            )
        if model_config.action_dim < 24:
            raise ValueError(
                f"latest-100 UMI hand contract requires model action_dim>=24, got {model_config.action_dim}"
            )
        if self.use_delta_eef_actions:
            raise ValueError(
                "UMI spatial actions must not use legacy DeltaEefEulerActions; "
                "set use_delta_eef_actions=False"
            )

        output_transforms: list[_transforms.DataTransformFn] = [
            cx002_policy.CX002Outputs(action_dim=24)
        ]
        if self.output_absolute_eef_actions:
            output_transforms.insert(0, _transforms.SpatialAbsoluteEefEulerActions12())

        data_transforms = _transforms.Group(
            inputs=[
                cx002_policy.CX002Inputs(
                    use_head_camera=self.use_head_camera,
                ),
                _transforms.SpatialDeltaEefEulerActions12(),
            ],
            outputs=output_transforms,
        )
        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)
        base_config = self.create_base_config(assets_dirs, model_config)
        if self.use_quantile_norm is not None:
            base_config = dataclasses.replace(base_config, use_quantile_norm=self.use_quantile_norm)
        return dataclasses.replace(
            base_config,
            repack_transforms=self.repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
            sample_filter_key=self.sample_filter_key,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotUmiPrechunkedEefRot6dHand30DataConfig(DataConfigFactory):
    """UMI state30 with a precomputed shared-anchor action chunk.

    State is adjacent body-frame delta EEF9 per arm followed by absolute hand6
    per arm. Each LeRobot row already stores action shape ``(H, 30)`` where
    all EEF targets use the current row as their common SE(3) anchor and hand
    targets remain absolute. Therefore the loader must not request future rows
    and no DeltaEef/AbsoluteEef transform may be applied.
    """

    use_head_camera: bool = True
    default_prompt: str | None = None
    use_delta_eef_actions: bool = False
    output_absolute_eef_actions: bool = False
    use_quantile_norm: bool | None = None
    sample_filter_key: str | None = None
    action_padding_mask_key: str | None = "action_is_pad"
    required_action_horizon: int = 50
    head_crop_normalized: tuple[float, float, float, float] | None = None
    head_crop_expected_aspect_ratio: float | None = None

    repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(
        default=_transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": {
                            "cam_high": "observation.images.head_rgb",
                            "cam_left_wrist": "observation.images.left_wrist_rgb",
                            "cam_right_wrist": "observation.images.right_wrist_rgb",
                        },
                        "state": "observation.state",
                        "actions": "action",
                        "action_is_pad": "action_is_pad",
                        "prompt": "prompt",
                    }
                )
            ]
        )
    )
    # The current LeRobot row already contains the entire (H,30) action chunk.
    action_sequence_keys: Sequence[str] = ()

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        if model_config.action_horizon != self.required_action_horizon:
            raise ValueError(
                "UMI prechunked contract requires "
                f"action_horizon={self.required_action_horizon}, got {model_config.action_horizon}"
            )
        if model_config.action_dim < 30:
            raise ValueError(
                f"UMI state/action contract requires model action_dim>=30, got {model_config.action_dim}"
            )
        if self.use_delta_eef_actions or self.output_absolute_eef_actions:
            raise ValueError(
                "UMI actions are already shared-anchor relative chunks; both "
                "use_delta_eef_actions and output_absolute_eef_actions must be False"
            )

        data_transforms = _transforms.Group(
            inputs=[
                cx002_policy.CX002Inputs(
                    use_head_camera=self.use_head_camera,
                    head_crop_normalized=self.head_crop_normalized,
                    head_crop_expected_aspect_ratio=self.head_crop_expected_aspect_ratio,
                )
            ],
            outputs=[cx002_policy.CX002Outputs(action_dim=30)],
        )
        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)
        base_config = self.create_base_config(assets_dirs, model_config)
        if self.use_quantile_norm is not None:
            base_config = dataclasses.replace(base_config, use_quantile_norm=self.use_quantile_norm)
        return dataclasses.replace(
            base_config,
            repack_transforms=self.repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
            sample_filter_key=self.sample_filter_key,
            action_padding_mask_key=self.action_padding_mask_key,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotLiberoDataConfig(DataConfigFactory):
    """
    This config is used to configure transforms that are applied at various parts of the data pipeline.
    For your own dataset, you can copy this class and modify the transforms to match your dataset based on the
    comments below.
    """

    extra_delta_transform: bool = False

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # The repack transform is *only* applied to the data coming from the dataset,
        # and *not* during inference. We can use it to make inputs from the dataset look
        # as close as possible to those coming from the inference environment (e.g. match the keys).
        # Below, we match the keys in the dataset (which we defined in the data conversion script) to
        # the keys we use in our inference pipeline (defined in the inference script for libero).
        # For your own dataset, first figure out what keys your environment passes to the policy server
        # and then modify the mappings below so your dataset's keys get matched to those target keys.
        # The repack transform simply remaps key names here.
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "image",
                        "observation/wrist_image": "wrist_image",
                        "observation/state": "state",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        # The data transforms are applied to the data coming from the dataset *and* during inference.
        # Below, we define the transforms for data going into the model (``inputs``) and the transforms
        # for data coming out of the model (``outputs``) (the latter is only used during inference).
        # We defined these transforms in `libero_policy.py`. You can check the detailed comments there for
        # how to modify the transforms to match your dataset. Once you created your own transforms, you can
        # replace the transforms below with your own.
        data_transforms = _transforms.Group(
            inputs=[libero_policy.LiberoInputs(model_type=model_config.model_type)],
            outputs=[libero_policy.LiberoOutputs()],
        )

        # One additional data transform: pi0 models are trained on delta actions (relative to the first
        # state in each action chunk). IF your data has ``absolute`` actions (e.g. target joint angles)
        # you can uncomment the following line to convert the actions to delta actions. The only exception
        # is for the gripper actions which are always absolute.
        # In the example below, we would apply the delta conversion to the first 6 actions (joints) and
        # leave the 7th action (gripper) unchanged, i.e. absolute.
        # In Libero, the raw actions in the dataset are already delta actions, so we *do not* need to
        # apply a separate delta conversion (that's why it's commented out). Choose whether to apply this
        # transform based on whether your dataset uses ``absolute`` or ``delta`` actions out of the box.

        # LIBERO already represents actions as deltas, but we have some old Pi0 checkpoints that are trained with this
        # extra delta transform.
        if self.extra_delta_transform:
            delta_action_mask = _transforms.make_bool_mask(6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        # Model transforms include things like tokenizing the prompt and action targets
        # You do not need to change anything here for your own dataset.
        model_transforms = ModelTransformFactory()(model_config)

        # We return all data transforms for training and inference. No need to change anything here.
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class RLDSDroidDataConfig(DataConfigFactory):
    """
    Config for training on DROID, using RLDS data format (for efficient training on larger datasets).
    """

    rlds_data_dir: str | None = None
    action_space: droid_rlds_dataset.DroidActionSpace | None = None

    # Filtering options. Can pass a path to a dictionary that maps episodes to timestep ranges
    # to tuples denoting ranges of time steps to keep (start, end). Episodes are uniquely identified with
    # f"{recording_folderpath}--{file_path}", both of which are present in the RLDS episode metadata.

    # List of datasets to sample from: name, version, weight, and optionally filter_dict_path
    datasets: Sequence[droid_rlds_dataset.RLDSDataset] = (
        droid_rlds_dataset.RLDSDataset(
            name="droid",
            version="1.0.1",
            weight=1.0,
            filter_dict_path="gs://openpi-assets/droid/droid_sample_ranges_v1_0_1.json",
        ),
    )

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/exterior_image_1_left": "observation/image",
                        "observation/wrist_image_left": "observation/wrist_image",
                        "observation/joint_position": "observation/joint_position",
                        "observation/gripper_position": "observation/gripper_position",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[droid_policy.DroidInputs(model_type=model_config.model_type)],
            outputs=[droid_policy.DroidOutputs()],
        )

        if self.action_space == droid_rlds_dataset.DroidActionSpace.JOINT_POSITION:
            # Data loader returns absolute joint position actions -- convert to delta actions for training.
            delta_action_mask = _transforms.make_bool_mask(7, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory()(model_config)

        assert self.rlds_data_dir is not None, "Need to set rlds data dir for RLDS data loader."

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            rlds_data_dir=self.rlds_data_dir,
            action_space=self.action_space,
            datasets=self.datasets,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotDROIDDataConfig(DataConfigFactory):
    """
    Example data config for custom DROID dataset in LeRobot format.
    To convert your custom DROID dataset (<10s of hours) to LeRobot format, see examples/droid/convert_droid_data_to_lerobot.py
    """

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/exterior_image_1_left": "exterior_image_1_left",
                        "observation/exterior_image_2_left": "exterior_image_2_left",
                        "observation/wrist_image_left": "wrist_image_left",
                        "observation/joint_position": "joint_position",
                        "observation/gripper_position": "gripper_position",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )
        # We assume joint *velocity* actions, so we should *not* apply an additional delta transform.
        data_transforms = _transforms.Group(
            inputs=[droid_policy.DroidInputs(model_type=model_config.model_type)],
            outputs=[droid_policy.DroidOutputs()],
        )
        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    # Name of the config. Must be unique. Will be used to reference this config.
    name: tyro.conf.Suppress[str]
    # Project name.
    project_name: str = "openpi"
    # Experiment name. Will be used to name the metadata and checkpoint directories.
    exp_name: str = tyro.MISSING

    # Defines the model config. Some attributes (action_dim, action_horizon, and max_token_len) are shared by all models
    # -- see BaseModelConfig. Specific model implementations (e.g., Pi0Config) inherit from BaseModelConfig and may
    # define additional attributes.
    model: _model.BaseModelConfig = dataclasses.field(default_factory=pi0_config.Pi0Config)

    # A weight loader can optionally load (possibly partial) weights from disk after the model is initialized.
    weight_loader: weight_loaders.WeightLoader = dataclasses.field(default_factory=weight_loaders.NoOpWeightLoader)

    # Optional path to a PyTorch checkpoint to load weights from.
    pytorch_weight_path: str | None = None

    # Precision for PyTorch training.
    pytorch_training_precision: Literal["bfloat16", "float32"] = "bfloat16"

    lr_schedule: _optimizer.LRScheduleConfig = dataclasses.field(default_factory=_optimizer.CosineDecaySchedule)
    optimizer: _optimizer.OptimizerConfig = dataclasses.field(default_factory=_optimizer.AdamW)
    ema_decay: float | None = 0.99

    # Specifies which weights should be frozen.
    freeze_filter: tyro.conf.Suppress[Filter] = dataclasses.field(default_factory=nnx.Nothing)

    # Determines the data to be trained on.
    data: DataConfigFactory = dataclasses.field(default_factory=FakeDataConfig)

    # Base directory for config assets (e.g., norm stats).
    assets_base_dir: str = "./assets"
    # Base directory for checkpoints.
    checkpoint_base_dir: str = "./checkpoints"

    # Random seed that will be used by random generators during training.
    seed: int = 42
    # Global batch size.
    batch_size: int = 32
    # Number of workers to use for the data loader. Increasing this number will speed up data loading but
    # will increase memory and CPU usage.
    num_workers: int = 2
    # Number of train steps (batches) to run.
    num_train_steps: int = 30_000

    # How often (in steps) to log training metrics.
    log_interval: int = 100
    # How often (in steps) to save checkpoints.
    save_interval: int = 1000
    # If set, any existing checkpoints matching step % keep_period == 0 will not be deleted.
    keep_period: int | None = 5000

    # If true, will overwrite the checkpoint directory if it already exists.
    overwrite: bool = False
    # If true, will resume training from the last checkpoint.
    resume: bool = False

    # If true, will enable wandb logging.
    wandb_enabled: bool = True

    # Used to pass metadata to the policy server.
    policy_metadata: dict[str, Any] | None = None

    # If the value is greater than 1, FSDP will be enabled and shard across number of specified devices; overall
    # device memory will be reduced but training could potentially be slower.
    # eg. if total device is 4 and fsdp devices is 2; then the model will shard to 2 devices and run
    # data parallel between 2 groups of devices.
    fsdp_devices: int = 1

    @property
    def assets_dirs(self) -> pathlib.Path:
        """Get the assets directory for this config."""
        return (pathlib.Path(self.assets_base_dir) / self.name).resolve()

    @property
    def checkpoint_dir(self) -> pathlib.Path:
        """Get the checkpoint directory for this config."""
        if not self.exp_name:
            raise ValueError("--exp_name must be set")
        return (pathlib.Path(self.checkpoint_base_dir) / self.name / self.exp_name).resolve()

    @property
    def trainable_filter(self) -> nnx.filterlib.Filter:
        """Get the filter for the trainable parameters."""
        return nnx.All(nnx.Param, nnx.Not(self.freeze_filter))

    def __post_init__(self) -> None:
        if self.resume and self.overwrite:
            raise ValueError("Cannot resume and overwrite at the same time.")


# Use `get_config` if you need to get a config by name in your code.
_CONFIGS = [
    #
    # Inference Aloha configs.
    #
    TrainConfig(
        name="pi0_aloha",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    TrainConfig(
        name="pi05_aloha",
        model=pi0_config.Pi0Config(pi05=True),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    TrainConfig(
        name="pi0_aloha_towel",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
            default_prompt="fold the towel",
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    TrainConfig(
        name="pi0_aloha_tupperware",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
            default_prompt="open the tupperware and put the food on the plate",
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    #
    # Inference DROID configs.
    #
    TrainConfig(
        name="pi0_droid",
        model=pi0_config.Pi0Config(action_horizon=10),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI0)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    TrainConfig(
        name="pi0_fast_droid",
        model=pi0_fast.Pi0FASTConfig(action_dim=8, action_horizon=10),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI0_FAST)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    TrainConfig(
        name="pi05_droid",
        model=pi0_config.Pi0Config(action_horizon=15, pi05=True),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI05)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    #
    # Fine-tuning Libero configs.
    #
    # These train configs define the hyperparameters for fine-tuning the base model on your own dataset.
    # They are used to define key elements like the dataset you are training on, the base checkpoint you
    # are using, and other hyperparameters like how many training steps to run or what learning rate to use.
    # For your own dataset, you can copy this class and modify the dataset name, and data transforms based on
    # the comments below.
    TrainConfig(
        # Change the name to reflect your model and dataset.
        name="pi0_libero",
        # Here you define the model config -- In this example we use pi0 as the model
        # architecture and perform *full* finetuning. in the examples below we show how to modify
        # this to perform *low-memory* (LORA) finetuning and use pi0-FAST as an alternative architecture.
        model=pi0_config.Pi0Config(),
        # Here you define the dataset you are training on. In this example we use the Libero
        # dataset. For your own dataset, you can change the repo_id to point to your dataset.
        # Also modify the DataConfig to use the new config you made for your dataset above.
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                # This flag determines whether we load the prompt (i.e. the task instruction) from the
                # ``task`` field in the LeRobot dataset. If set to True, the prompt will show up in
                # a field called ``prompt`` in the input dict. The recommended setting is True.
                prompt_from_task=True,
            ),
            extra_delta_transform=True,
        ),
        # Here you define which pre-trained checkpoint you want to load to initialize the model.
        # This should match the model config you chose above -- i.e. in this case we use the pi0 base model.
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        # Below you can define other hyperparameters like the learning rate, number of training steps, etc.
        # Check the base TrainConfig class for a full list of available hyperparameters.
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi0_libero_low_mem_finetune",
        # Here is an example of loading a pi0 model for LoRA fine-tuning.
        model=pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=30_000,
        # The freeze filter defines which parameters should be frozen during training.
        # We have a convenience function in the model config that returns the default freeze filter
        # for the given model config for LoRA finetuning. Just make sure it matches the model config
        # you chose above.
        freeze_filter=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
        # Turn off EMA for LoRA finetuning.
        ema_decay=None,
    ),
    TrainConfig(
        name="pi0_fast_libero",
        # Here is an example of loading a pi0-FAST model for full finetuning.
        # Modify action_dim and action_horizon to match your dataset (action horizon is equal to
        # the desired action chunk length).
        # The max_token_len is the maximum number of (non-image) tokens the model can handle.
        # This includes the tokenized prompt, proprioceptive state, and (FAST-tokenized) action tokens.
        # Choosing this value too small may chop off tokens at the end of your sequence (the code will throw
        # a warning), while choosing it too large will waste memory (since we pad each batch element to the
        # max_token_len). A good rule of thumb is to use approx 180 for single-arm robots, and approx 250 for
        # two-arm robots. Generally, err on the lower side here first, and potentially increase the value if
        # you see many warnings being thrown during training.
        model=pi0_fast.Pi0FASTConfig(action_dim=7, action_horizon=10, max_token_len=180),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ),
        # Note that we load the pi0-FAST base model checkpoint here.
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi0_fast_libero_low_mem_finetune",
        # Here is an example of loading a pi0-FAST model for LoRA finetuning.
        # For setting action_dim, action_horizon, and max_token_len, see the comments above.
        model=pi0_fast.Pi0FASTConfig(
            action_dim=7, action_horizon=10, max_token_len=180, paligemma_variant="gemma_2b_lora"
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=30_000,
        # Again, make sure to match the model config above when extracting the freeze filter
        # that specifies which parameters should be frozen during LoRA finetuning.
        freeze_filter=pi0_fast.Pi0FASTConfig(
            action_dim=7, action_horizon=10, max_token_len=180, paligemma_variant="gemma_2b_lora"
        ).get_freeze_filter(),
        # Turn off EMA for LoRA finetuning.
        ema_decay=None,
    ),
    TrainConfig(
        name="pi05_libero",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),
        batch_size=256,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path="/path/to/your/pytorch_weight_path",
        num_train_steps=30_000,
    ),
    #
    # CX002 task2 pi0.5 relative dual-EEF fine-tuning. The dataset stores
    # 14D absolute xyz+rpy EEF state/action vectors and the data config
    # converts future targets to current-state-anchored relative actions.
    #
    TrainConfig(
        name="pi05_cx002_relative_eef",
        assets_base_dir="/mnt/data/dzq/openpi/data/assets",
        checkpoint_base_dir="/mnt/data/dzq/openpi/checkpoints",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=30,
            discrete_state_input=False,
        ),
        data=LeRobotCX002RawEef6DataConfig(
            repo_id="/mnt/data/dzq/openpi/datasets/task2_eef6",
            assets=AssetsConfig(asset_id="task2_eef6"),
            base_config=DataConfig(prompt_from_task=True),
            default_prompt=_CX002_TASK2_PROMPT,
            use_delta_eef_actions=True,
            output_absolute_eef_actions=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("/mnt/data/checkpoints/pi05_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=500,
            peak_lr=2.5e-5,
            decay_steps=30_000,
            decay_lr=2.5e-6,
        ),
        ema_decay=None,
        batch_size=64,
        num_workers=8,
        num_train_steps=30_000,
        log_interval=100,
        save_interval=1_000,
        keep_period=5_000,
        wandb_enabled=False,
        fsdp_devices=8,
    ),
    # Latest-100 UMI: 10 Hz continuous state30 and one precomputed H50x30
    # shared-anchor relative action chunk in every LeRobot row.
    TrainConfig(
        name="pi05_umi_latest100_delta_eef_rot6d_hand30_10hz_h50",
        assets_base_dir="/mnt/data/dzq/openpi/data/assets",
        checkpoint_base_dir="/mnt/data/dzq/openpi/checkpoints",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            discrete_state_input=True,
        ),
        data=LeRobotUmiPrechunkedEefRot6dHand30DataConfig(
            repo_id="/mnt/data/dzq/umi/data/最新100条_lerobot30",
            assets=AssetsConfig(asset_id="umi_latest100_delta_eef_rot6d_hand30_10hz_h50"),
            base_config=DataConfig(prompt_from_task=True),
            default_prompt=_UMI_BOX_IN_OUT_PROMPT,
            use_delta_eef_actions=False,
            output_absolute_eef_actions=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/mnt/data/checkpoints/pi05_base/params"
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=500,
            peak_lr=2.5e-5,
            decay_steps=20_000,
            decay_lr=2.5e-6,
        ),
        batch_size=64,
        num_workers=6,
        num_train_steps=20_000,
        log_interval=50,
        save_interval=2_000,
        keep_period=2_000,
        wandb_enabled=False,
        fsdp_devices=8,
    ),
    # No-tail UMI v2: all 10 Hz observation/video rows are retained.  Only
    # anchors with 50 real future targets are exposed to training; terminal
    # padded rows remain in LeRobot for complete episode playback/auditing.
    TrainConfig(
        name="pi05_umi_taskumi1_no_tail_10hz_h50_v2",
        assets_base_dir="/mnt/data/dzq/openpi/data/assets",
        checkpoint_base_dir="/mnt/data/dzq/openpi/checkpoints",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            discrete_state_input=True,
        ),
        data=LeRobotUmiPrechunkedEefRot6dHand30DataConfig(
            repo_id="/mnt/data/dzq/umi/datasets/taskumi1_no_tail",
            assets=AssetsConfig(asset_id="umi_taskumi1_no_tail_10hz_h50_v2"),
            base_config=DataConfig(prompt_from_task=True),
            default_prompt=_UMI_BOX_IN_OUT_PROMPT,
            use_delta_eef_actions=False,
            output_absolute_eef_actions=False,
            sample_filter_key="sample_valid_h50",
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/mnt/data/checkpoints/pi05_base/params"
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=500,
            peak_lr=2.5e-5,
            decay_steps=20_000,
            decay_lr=2.5e-6,
        ),
        batch_size=64,
        num_workers=6,
        num_train_steps=20_000,
        log_interval=50,
        save_interval=2_000,
        keep_period=2_000,
        wandb_enabled=False,
        fsdp_devices=8,
    ),
    # task_v1 native-layout conversion. Every real terminal anchor remains in
    # the dataset; action_is_pad is propagated to the JAX loss and masks only
    # padded future slots (no sample_valid_h50 row filtering).
    TrainConfig(
        name="pi05_umi_task_v1_hand_pose_10hz_h50_masked_with_head_v1",
        assets_base_dir="/mnt/data/dzq/openpi/data/assets",
        checkpoint_base_dir="/mnt/data/dzq/openpi/checkpoints",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            discrete_state_input=True,
        ),
        data=LeRobotUmiPrechunkedEefRot6dHand30DataConfig(
            repo_id="/mnt/data/dzq/umi/datasets/task_v1_lerobot_h50",
            assets=AssetsConfig(
                assets_dir="/mnt/data/dzq/openpi/data/assets",
                asset_id="umi_task_v1_hand_pose_10hz_h50_masked_v1",
            ),
            base_config=DataConfig(prompt_from_task=True),
            default_prompt=_UMI_TASK_V1_PROMPT,
            use_head_camera=True,
            use_delta_eef_actions=False,
            output_absolute_eef_actions=False,
            sample_filter_key=None,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/mnt/data/checkpoints/pi05_base/params"
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=500,
            peak_lr=2.5e-5,
            decay_steps=20_000,
            decay_lr=2.5e-6,
        ),
        batch_size=64,
        num_workers=6,
        num_train_steps=20_000,
        log_interval=50,
        save_interval=2_000,
        keep_period=2_000,
        wandb_enabled=False,
        fsdp_devices=8,
    ),
    # task_v1 training config with the user-marked table ROI applied only to
    # the head camera before ResizeImages(224, 224).
    TrainConfig(
        name="pi05_umi_task_v1_hand_pose_10hz_h50_masked_with_head_roi_v1",
        assets_base_dir="/mnt/data/dzq/openpi/data/assets",
        checkpoint_base_dir="/mnt/data/dzq/openpi/checkpoints",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            discrete_state_input=True,
        ),
        data=LeRobotUmiPrechunkedEefRot6dHand30DataConfig(
            repo_id="/mnt/data/dzq/umi/datasets/task_v1_lerobot_h50",
            assets=AssetsConfig(
                assets_dir="/mnt/data/dzq/openpi/data/assets",
                asset_id="umi_task_v1_hand_pose_10hz_h50_masked_v1",
            ),
            base_config=DataConfig(prompt_from_task=True),
            default_prompt=_UMI_TASK_V1_PROMPT,
            use_head_camera=True,
            head_crop_normalized=cx002_policy.TASK_V1_HEAD_CROP_NORMALIZED,
            head_crop_expected_aspect_ratio=cx002_policy.TASK_V1_HEAD_INPUT_ASPECT_RATIO,
            use_delta_eef_actions=False,
            output_absolute_eef_actions=False,
            sample_filter_key=None,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/mnt/data/checkpoints/pi05_base/params"
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=500,
            peak_lr=2.5e-5,
            decay_steps=20_000,
            decay_lr=2.5e-6,
        ),
        batch_size=64,
        num_workers=6,
        num_train_steps=20_000,
        log_interval=50,
        save_interval=2_000,
        keep_period=2_000,
        wandb_enabled=False,
        policy_metadata={
            "task_prompt": _UMI_TASK_V1_PROMPT,
            "image_mode": "training-roi-cropped",
            "head_crop_normalized_ltrb": cx002_policy.TASK_V1_HEAD_CROP_NORMALIZED,
            "head_crop_coordinate_order": "left,top,right,bottom",
            "head_crop_expected_input_aspect": "4:3",
            "head_crop_applied_by": "CX002Inputs-before-ResizeImages",
        },
        fsdp_devices=8,
    ),
    # task_v1_new 10 Hz/H50 10k-step training run. This deliberately reuses the
    # task_v1 ROI/model/optimizer/batch/FSDP contract while keeping dataset
    # assets and checkpoints isolated from the earlier task_v1 experiment.
    TrainConfig(
        name="pi05_umi_task_v1_new_10hz_h50_masked_with_head_roi_10k_w32_v1",
        assets_base_dir="/mnt/data/dzq/openpi/data/assets",
        checkpoint_base_dir="/mnt/data/dzq/openpi/checkpoints",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            discrete_state_input=True,
        ),
        data=LeRobotUmiPrechunkedEefRot6dHand30DataConfig(
            repo_id="/mnt/data/dzq/umi/datasets/task_v1_new_lerobot_10hz_h50",
            assets=AssetsConfig(
                assets_dir="/mnt/data/dzq/openpi/data/assets",
                asset_id="umi_task_v1_new_hand_pose_10hz_h50_masked_v1",
            ),
            base_config=DataConfig(prompt_from_task=True),
            default_prompt=_UMI_TASK_V1_NEW_PROMPT,
            use_head_camera=True,
            head_crop_normalized=cx002_policy.TASK_V1_HEAD_CROP_NORMALIZED,
            head_crop_expected_aspect_ratio=cx002_policy.TASK_V1_HEAD_INPUT_ASPECT_RATIO,
            use_delta_eef_actions=False,
            output_absolute_eef_actions=False,
            sample_filter_key=None,
            action_padding_mask_key="action_is_pad",
            required_action_horizon=50,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/mnt/data/checkpoints/pi05_base/params"
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=500,
            peak_lr=2.5e-5,
            decay_steps=20_000,
            decay_lr=2.5e-6,
        ),
        batch_size=64,
        num_workers=32,
        num_train_steps=10_000,
        log_interval=50,
        save_interval=1_000,
        keep_period=1_000,
        wandb_enabled=False,
        policy_metadata={
            "task_prompt": _UMI_TASK_V1_NEW_PROMPT,
            "dataset_fps": 10,
            "action_horizon": 50,
            "action_horizon_seconds": 5.0,
            "image_mode": "training-roi-cropped",
            "head_crop_normalized_ltrb": cx002_policy.TASK_V1_HEAD_CROP_NORMALIZED,
            "head_crop_coordinate_order": "left,top,right,bottom",
            "head_crop_expected_input_aspect": "4:3",
            "head_crop_applied_by": "CX002Inputs-before-ResizeImages",
            "action_padding_mask_key": "action_is_pad",
        },
        fsdp_devices=8,
    ),
    # task_v1_new 10 Hz/H50 full-head ablation. This is identical to the ROI
    # run except that the complete 4:3 head frame is kept before the shared
    # aspect-preserving ResizeImages(224, 224) transform.
    TrainConfig(
        name="pi05_umi_task_v1_new_10hz_h50_masked_with_full_head_10k_w32_v1",
        assets_base_dir="/mnt/data/dzq/openpi/data/assets",
        checkpoint_base_dir="/mnt/data/dzq/openpi/checkpoints",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            discrete_state_input=True,
        ),
        data=LeRobotUmiPrechunkedEefRot6dHand30DataConfig(
            repo_id="/mnt/data/dzq/umi/datasets/task_v1_new_lerobot_10hz_h50",
            assets=AssetsConfig(
                assets_dir="/mnt/data/dzq/openpi/data/assets",
                asset_id="umi_task_v1_new_hand_pose_10hz_h50_masked_v1",
            ),
            base_config=DataConfig(prompt_from_task=True),
            default_prompt=_UMI_TASK_V1_NEW_PROMPT,
            use_head_camera=True,
            head_crop_normalized=None,
            head_crop_expected_aspect_ratio=cx002_policy.TASK_V1_HEAD_INPUT_ASPECT_RATIO,
            use_delta_eef_actions=False,
            output_absolute_eef_actions=False,
            sample_filter_key=None,
            action_padding_mask_key="action_is_pad",
            required_action_horizon=50,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/mnt/data/checkpoints/pi05_base/params"
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=500,
            peak_lr=2.5e-5,
            decay_steps=20_000,
            decay_lr=2.5e-6,
        ),
        batch_size=64,
        num_workers=32,
        num_train_steps=10_000,
        log_interval=50,
        save_interval=1_000,
        keep_period=1_000,
        wandb_enabled=False,
        policy_metadata={
            "task_prompt": _UMI_TASK_V1_NEW_PROMPT,
            "dataset_fps": 10,
            "action_horizon": 50,
            "action_horizon_seconds": 5.0,
            "image_mode": "training-full-frame-uncropped",
            "head_crop_enabled": False,
            "head_input_expected_aspect": "4:3",
            "head_resize": "ResizeImages-224x224-with-pad",
            "action_padding_mask_key": "action_is_pad",
        },
        fsdp_devices=8,
    ),
    # task_v3 10 Hz/H30 full-head training. Actions are prechunked in the
    # LeRobot rows; action_is_pad masks only terminal padding slots.
    TrainConfig(
        name="pi05_umi_task_v3_10hz_h30_masked_with_full_head_10k_w32_v1",
        assets_base_dir="/mnt/data/dzq/openpi/data/assets",
        checkpoint_base_dir="/mnt/data/dzq/openpi/checkpoints",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=30,
            discrete_state_input=True,
        ),
        data=LeRobotUmiPrechunkedEefRot6dHand30DataConfig(
            repo_id="/mnt/data/dzq/umi/datasets/task_v3_lerobot_10hz_h30",
            assets=AssetsConfig(
                assets_dir="/mnt/data/dzq/openpi/data/assets",
                asset_id="umi_task_v3_hand_pose_10hz_h30_masked_v1",
            ),
            base_config=DataConfig(prompt_from_task=True),
            default_prompt=_UMI_TASK_V3_PROMPT,
            use_head_camera=True,
            head_crop_normalized=None,
            head_crop_expected_aspect_ratio=cx002_policy.TASK_V1_HEAD_INPUT_ASPECT_RATIO,
            use_delta_eef_actions=False,
            output_absolute_eef_actions=False,
            sample_filter_key=None,
            action_padding_mask_key="action_is_pad",
            required_action_horizon=30,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/mnt/data/checkpoints/pi05_base/params"
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=500,
            peak_lr=2.5e-5,
            decay_steps=20_000,
            decay_lr=2.5e-6,
        ),
        batch_size=32,
        num_workers=32,
        num_train_steps=10_000,
        log_interval=50,
        save_interval=1_000,
        keep_period=1_000,
        wandb_enabled=False,
        policy_metadata={
            "task_prompt": _UMI_TASK_V3_PROMPT,
            "dataset_fps": 10,
            "action_horizon": 30,
            "action_horizon_seconds": 3.0,
            "image_mode": "training-full-frame-uncropped",
            "head_crop_enabled": False,
            "head_input_expected_aspect": "4:3",
            "head_resize": "ResizeImages-224x224-with-pad",
            "action_padding_mask_key": "action_is_pad",
        },
        fsdp_devices=4,
    ),
    # task_v3 10 Hz/H50 full-head training. The dataset already stores each
    # H50 action chunk; action_is_pad masks only terminal padding slots.
    TrainConfig(
        name="pi05_umi_task_v3_10hz_h50_masked_with_full_head_10k_w32_v1",
        assets_base_dir="/mnt/data/dzq/openpi/data/assets",
        checkpoint_base_dir="/mnt/data/dzq/openpi/checkpoints",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            discrete_state_input=True,
        ),
        data=LeRobotUmiPrechunkedEefRot6dHand30DataConfig(
            repo_id="/mnt/data/dzq/umi/datasets/task_v3_lerobot_10hz_h50",
            assets=AssetsConfig(
                assets_dir="/mnt/data/dzq/openpi/data/assets",
                asset_id="umi_task_v3_hand_pose_10hz_h50_masked_v1",
            ),
            base_config=DataConfig(prompt_from_task=True),
            default_prompt=_UMI_TASK_V3_PROMPT,
            use_head_camera=True,
            head_crop_normalized=None,
            head_crop_expected_aspect_ratio=cx002_policy.TASK_V1_HEAD_INPUT_ASPECT_RATIO,
            use_delta_eef_actions=False,
            output_absolute_eef_actions=False,
            sample_filter_key=None,
            action_padding_mask_key="action_is_pad",
            required_action_horizon=50,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/mnt/data/checkpoints/pi05_base/params"
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=500,
            peak_lr=2.5e-5,
            decay_steps=20_000,
            decay_lr=2.5e-6,
        ),
        batch_size=32,
        num_workers=32,
        num_train_steps=10_000,
        log_interval=50,
        save_interval=1_000,
        keep_period=1_000,
        wandb_enabled=False,
        policy_metadata={
            "task_prompt": _UMI_TASK_V3_PROMPT,
            "dataset_fps": 10,
            "action_horizon": 50,
            "action_horizon_seconds": 5.0,
            "image_mode": "training-full-frame-uncropped",
            "head_crop_enabled": False,
            "head_input_expected_aspect": "4:3",
            "head_resize": "ResizeImages-224x224-with-pad",
            "action_padding_mask_key": "action_is_pad",
        },
        fsdp_devices=4,
    ),
    # task_v3_high is the explicit 51-episode quality allowlist from task_v3,
    # converted independently at 10 Hz/H50.  Pi0.5 tokenizes the normalized
    # proprioceptive state only when discrete_state_input=True.
    TrainConfig(
        name="pi05_umi_task_v3_high_10hz_h50_masked_with_full_head_state_10k_b64_w32_8gpu_v1",
        assets_base_dir="/mnt/data/dzq/openpi/data/assets",
        checkpoint_base_dir="/mnt/data/dzq/openpi/checkpoints",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            discrete_state_input=True,
        ),
        data=LeRobotUmiPrechunkedEefRot6dHand30DataConfig(
            repo_id="/mnt/data/dzq/umi/datasets/task_v3_high_lerobot_10hz_h50",
            assets=AssetsConfig(
                assets_dir="/mnt/data/dzq/openpi/data/assets",
                asset_id="umi_task_v3_high_hand_pose_10hz_h50_masked_v1",
            ),
            base_config=DataConfig(prompt_from_task=True),
            default_prompt=_UMI_TASK_V3_PROMPT,
            use_head_camera=True,
            head_crop_normalized=None,
            head_crop_expected_aspect_ratio=cx002_policy.TASK_V1_HEAD_INPUT_ASPECT_RATIO,
            use_delta_eef_actions=False,
            output_absolute_eef_actions=False,
            sample_filter_key=None,
            action_padding_mask_key="action_is_pad",
            required_action_horizon=50,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/mnt/data/checkpoints/pi05_base/params"
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=500,
            peak_lr=2.5e-5,
            decay_steps=20_000,
            decay_lr=2.5e-6,
        ),
        batch_size=64,
        num_workers=32,
        num_train_steps=10_000,
        log_interval=50,
        save_interval=1_000,
        keep_period=1_000,
        wandb_enabled=False,
        policy_metadata={
            "task_prompt": _UMI_TASK_V3_PROMPT,
            "dataset_variant": "task_v3_high",
            "selected_source_episodes": 51,
            "episode_selection_sha256": "89f570b8e22344bb7a46a728b8460aacc831afccec5b9e22855fd52e5d278b32",
            "dataset_fps": 10,
            "action_horizon": 50,
            "action_horizon_seconds": 5.0,
            "state_input_mode": "pi05_discrete_tokens",
            "discrete_state_input": True,
            "image_mode": "training-full-frame-uncropped",
            "head_crop_enabled": False,
            "head_input_expected_aspect": "4:3",
            "head_resize": "ResizeImages-224x224-with-pad",
            "action_padding_mask_key": "action_is_pad",
        },
        fsdp_devices=8,
    ),
    # Quality-selected task_v2_x2: same Pi0.5 H50 recipe, with its own
    # dataset, task instruction and normalization asset.
    TrainConfig(
        name="pi05_umi_task_v2_x2_high_10hz_h50_masked_with_full_head_state_10k_b64_w32_8gpu_v1",
        assets_base_dir="/mnt/data/dzq/openpi/data/assets",
        checkpoint_base_dir="/mnt/data/dzq/openpi/checkpoints",
        model=pi0_config.Pi0Config(
            pi05=True, action_dim=32, action_horizon=50, discrete_state_input=True,
        ),
        data=LeRobotUmiPrechunkedEefRot6dHand30DataConfig(
            repo_id="/mnt/data/dzq/umi_v2/datasets/task_v2_x2_high_lerobot_10hz_h50",
            assets=AssetsConfig(
                assets_dir="/mnt/data/dzq/openpi/data/assets",
                asset_id="umi_task_v2_x2_high_hand_pose_10hz_h50_masked_v1",
            ),
            base_config=DataConfig(prompt_from_task=True),
            default_prompt="Put the two objects into the box.",
            use_head_camera=True,
            head_crop_normalized=None,
            head_crop_expected_aspect_ratio=cx002_policy.TASK_V1_HEAD_INPUT_ASPECT_RATIO,
            use_delta_eef_actions=False,
            output_absolute_eef_actions=False,
            sample_filter_key=None,
            action_padding_mask_key="action_is_pad",
            required_action_horizon=50,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/mnt/data/checkpoints/pi05_base/params"
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=500, peak_lr=2.5e-5, decay_steps=20_000, decay_lr=2.5e-6,
        ),
        batch_size=64,
        num_workers=32,
        num_train_steps=10_000,
        log_interval=50,
        save_interval=1_000,
        keep_period=1_000,
        wandb_enabled=False,
        policy_metadata={
            "task_prompt": "Put the two objects into the box.",
            "dataset_variant": "task_v2_x2_high",
            "requested_source_episodes": 95,
            "selected_source_episodes": 85,
            "excluded_source_episodes": 10,
            "episode_selection_sha256": "10529d85a309173ca01348bb092ca36278d8064737656b88994baa90152f4b2b",
            "original_episode_selection_sha256": "b6a05245e6b0eaf68746fa43851db2f2b097a20f87e0131444f692082a4ebffd",
            "discontinuity_policy": "exclude_10_whole_episodes_no_splitting",
            "dataset_fps": 10,
            "action_horizon": 50,
            "action_horizon_seconds": 5.0,
            "state_input_mode": "pi05_discrete_tokens",
            "discrete_state_input": True,
            "image_mode": "training-full-frame-uncropped",
            "head_crop_enabled": False,
            "head_input_expected_aspect": "4:3",
            "head_resize": "ResizeImages-224x224-with-pad",
            "action_padding_mask_key": "action_is_pad",
        },
        fsdp_devices=8,
    ),
    # Corrected complete E6 right eye: source streams are decoded/probed before
    # cropping. This is a fresh run, not a resume from the old miscropped images.
    TrainConfig(
        name="pi05_umi_task_v2_x2_high_10hz_h50_right_eye_state_10k_b64_w32_8gpu_v2",
        assets_base_dir="/mnt/data/dzq/openpi/data/assets",
        checkpoint_base_dir="/mnt/data/dzq/openpi/checkpoints",
        model=pi0_config.Pi0Config(
            pi05=True, action_dim=32, action_horizon=50, discrete_state_input=True,
        ),
        data=LeRobotUmiPrechunkedEefRot6dHand30DataConfig(
            repo_id="/mnt/data/dzq/umi_v2/datasets/task_v2_x2_high_lerobot_10hz_h50_right_eye_v2",
            assets=AssetsConfig(
                assets_dir="/mnt/data/dzq/openpi/data/assets",
                asset_id="umi_task_v2_x2_high_hand_pose_10hz_h50_right_eye_masked_v2",
            ),
            base_config=DataConfig(prompt_from_task=True),
            default_prompt="Put the two objects into the box.",
            use_head_camera=True,
            head_crop_normalized=None,
            head_crop_expected_aspect_ratio=cx002_policy.TASK_V1_HEAD_INPUT_ASPECT_RATIO,
            use_delta_eef_actions=False,
            output_absolute_eef_actions=False,
            sample_filter_key=None,
            action_padding_mask_key="action_is_pad",
            required_action_horizon=50,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/mnt/data/checkpoints/pi05_base/params"
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=500, peak_lr=2.5e-5, decay_steps=20_000, decay_lr=2.5e-6,
        ),
        batch_size=64,
        num_workers=32,
        num_train_steps=10_000,
        log_interval=50,
        save_interval=1_000,
        keep_period=1_000,
        wandb_enabled=False,
        policy_metadata={
            "task_prompt": "Put the two objects into the box.",
            "dataset_variant": "task_v2_x2_high_right_eye_v2",
            "selected_source_episodes": 85,
            "excluded_source_episodes": 10,
            "episode_selection_sha256": "10529d85a309173ca01348bb092ca36278d8064737656b88994baa90152f4b2b",
            "original_episode_selection_sha256": "b6a05245e6b0eaf68746fa43851db2f2b097a20f87e0131444f692082a4ebffd",
            "discontinuity_policy": "exclude_10_whole_episodes_no_splitting",
            "dataset_fps": 10,
            "action_horizon": 50,
            "action_horizon_seconds": 5.0,
            "state_input_mode": "pi05_discrete_tokens",
            "discrete_state_input": True,
            "image_mode": "complete-right-eye-letterbox-no-fixed-roi",
            "head_crop_enabled": False,
            "head_source_actual_wh": [3840, 1200],
            "head_eye_crop_xywh": [1920, 0, 1920, 1200],
            "head_eye_selection": "decoded_width_right_half",
            "head_video_wh": [640, 480],
            "head_input_expected_aspect": "4:3",
            "head_resize": "ResizeImages-224x224-with-pad",
            "training_augmentation": "unchanged pi05 head 95pct random crop, +/-5deg rotation, color jitter",
            "action_padding_mask_key": "action_is_pad",
            "fresh_base_model": True,
        },
        fsdp_devices=8,
    ),
    # No-state ablation of the complete-right-eye task_v2_x2 run. Keep the
    # dataset, action targets, normalization, optimizer and base weights identical.
    # Pi05 has no continuous state projection; disabling tokenization removes
    # state values from model conditioning while retaining the interface field.
    TrainConfig(
        name="pi05_umi_task_v2_x2_high_10hz_h50_right_eye_no_state_10k_b64_w32_8gpu_v1",
        assets_base_dir="/mnt/data/dzq/openpi/data/assets",
        checkpoint_base_dir="/mnt/data/dzq/openpi/checkpoints",
        model=pi0_config.Pi0Config(
            pi05=True, action_dim=32, action_horizon=50, discrete_state_input=False,
        ),
        data=LeRobotUmiPrechunkedEefRot6dHand30DataConfig(
            repo_id="/mnt/data/dzq/umi_v2/datasets/task_v2_x2_high_lerobot_10hz_h50_right_eye_v2",
            assets=AssetsConfig(
                assets_dir="/mnt/data/dzq/openpi/data/assets",
                asset_id="umi_task_v2_x2_high_hand_pose_10hz_h50_right_eye_masked_v2",
            ),
            base_config=DataConfig(prompt_from_task=True),
            default_prompt="Put the two objects into the box.",
            use_head_camera=True,
            head_crop_normalized=None,
            head_crop_expected_aspect_ratio=cx002_policy.TASK_V1_HEAD_INPUT_ASPECT_RATIO,
            use_delta_eef_actions=False,
            output_absolute_eef_actions=False,
            sample_filter_key=None,
            action_padding_mask_key="action_is_pad",
            required_action_horizon=50,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/mnt/data/checkpoints/pi05_base/params"
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=500, peak_lr=2.5e-5, decay_steps=20_000, decay_lr=2.5e-6,
        ),
        batch_size=64,
        num_workers=32,
        num_train_steps=10_000,
        log_interval=50,
        save_interval=1_000,
        keep_period=1_000,
        wandb_enabled=False,
        policy_metadata={
            "task_prompt": "Put the two objects into the box.",
            "dataset_variant": "task_v2_x2_high_right_eye_v2",
            "selected_source_episodes": 85,
            "excluded_source_episodes": 10,
            "episode_selection_sha256": "10529d85a309173ca01348bb092ca36278d8064737656b88994baa90152f4b2b",
            "original_episode_selection_sha256": "b6a05245e6b0eaf68746fa43851db2f2b097a20f87e0131444f692082a4ebffd",
            "discontinuity_policy": "exclude_10_whole_episodes_no_splitting",
            "dataset_fps": 10,
            "action_horizon": 50,
            "action_horizon_seconds": 5.0,
            "state_input_mode": "none",
            "discrete_state_input": False,
            "state_values_used_for_conditioning": False,
            "state_field_retained_for_interface": True,
            "state_ablation_baseline_config": "pi05_umi_task_v2_x2_high_10hz_h50_right_eye_state_10k_b64_w32_8gpu_v2",
            "image_mode": "complete-right-eye-letterbox-no-fixed-roi",
            "head_crop_enabled": False,
            "head_source_actual_wh": [3840, 1200],
            "head_eye_crop_xywh": [1920, 0, 1920, 1200],
            "head_eye_selection": "decoded_width_right_half",
            "head_video_wh": [640, 480],
            "head_input_expected_aspect": "4:3",
            "head_resize": "ResizeImages-224x224-with-pad",
            "training_augmentation": "unchanged pi05 head 95pct random crop, +/-5deg rotation, color jitter",
            "action_padding_mask_key": "action_is_pad",
            "fresh_base_model": True,
        },
        fsdp_devices=8,
    ),
    # Each wrist is already rotated 180 degrees in this new dataset.
    # No additional training rotation; raw inference images need the matching adapter.
    TrainConfig(
        name="pi05_umi_task_v2_x2_high_10hz_h50_wrist180_no_state_10k_b64_w32_8gpu_v1",
        assets_base_dir="/mnt/data/dzq/openpi/data/assets",
        checkpoint_base_dir="/mnt/data/dzq/openpi/checkpoints",
        model=pi0_config.Pi0Config(
            pi05=True, action_dim=32, action_horizon=50, discrete_state_input=False,
        ),
        data=LeRobotUmiPrechunkedEefRot6dHand30DataConfig(
            repo_id="/mnt/data/dzq/umi_v2/datasets/task_v2_x2_high_lerobot_10hz_h50_right_eye_v2_wrist180",
            assets=AssetsConfig(
                assets_dir="/mnt/data/dzq/openpi/data/assets",
                asset_id="umi_task_v2_x2_high_hand_pose_10hz_h50_right_eye_wrist180_masked_v1",
            ),
            base_config=DataConfig(prompt_from_task=True),
            default_prompt="Put the two objects into the box.",
            use_head_camera=True,
            head_crop_normalized=None,
            head_crop_expected_aspect_ratio=cx002_policy.TASK_V1_HEAD_INPUT_ASPECT_RATIO,
            use_delta_eef_actions=False,
            output_absolute_eef_actions=False,
            sample_filter_key=None,
            action_padding_mask_key="action_is_pad",
            required_action_horizon=50,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/mnt/data/checkpoints/pi05_base/params"
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=500, peak_lr=2.5e-5, decay_steps=20_000, decay_lr=2.5e-6,
        ),
        batch_size=64,
        num_workers=32,
        num_train_steps=10_000,
        log_interval=50,
        save_interval=1_000,
        keep_period=1_000,
        wandb_enabled=False,
        policy_metadata={
            "task_prompt": "Put the two objects into the box.",
            "dataset_variant": "task_v2_x2_high_right_eye_v2_wrist180",
            "selected_source_episodes": 85,
            "excluded_source_episodes": 10,
            "episode_selection_sha256": "10529d85a309173ca01348bb092ca36278d8064737656b88994baa90152f4b2b",
            "original_episode_selection_sha256": "b6a05245e6b0eaf68746fa43851db2f2b097a20f87e0131444f692082a4ebffd",
            "discontinuity_policy": "exclude_10_whole_episodes_no_splitting",
            "dataset_fps": 10,
            "action_horizon": 50,
            "action_horizon_seconds": 5.0,
            "state_input_mode": "none",
            "discrete_state_input": False,
            "state_values_used_for_conditioning": False,
            "state_field_retained_for_interface": True,
            "state_ablation_baseline_config": "pi05_umi_task_v2_x2_high_10hz_h50_right_eye_state_10k_b64_w32_8gpu_v2",
            "image_mode": "complete-right-eye-letterbox-no-fixed-roi",
            "head_crop_enabled": False,
            "head_source_actual_wh": [3840, 1200],
            "head_eye_crop_xywh": [1920, 0, 1920, 1200],
            "head_eye_selection": "decoded_width_right_half",
            "head_video_wh": [640, 480],
            "head_input_expected_aspect": "4:3",
            "head_resize": "ResizeImages-224x224-with-pad",
            "training_augmentation": "unchanged pi05 head 95pct random crop, +/-5deg rotation, color jitter",
            "action_padding_mask_key": "action_is_pad",
            "fresh_base_model": True,
            "wrist_rotation_degrees": {"left": 180, "right": 180},
            "training_wrist_rotation_source": "encoded_dataset",
            "additional_training_wrist_rotation": False,
            "inference_wrist_rotation_required": True,
            "rotation_baseline_config": "pi05_umi_task_v2_x2_high_10hz_h50_right_eye_no_state_10k_b64_w32_8gpu_v1",
        },
        fsdp_devices=8,
    ),
    # Controlled state-input ablation: two disjoint four-GPU jobs, batch 32 each.
    TrainConfig(
        name="pi05_umi_task_v2_x2_high_10hz_h50_wrist180_no_state_10k_b32_w32_4gpu_v1",
        assets_base_dir="/mnt/data/dzq/openpi/data/assets",
        checkpoint_base_dir="/mnt/data/dzq/openpi/checkpoints",
        model=pi0_config.Pi0Config(
            pi05=True, action_dim=32, action_horizon=50, discrete_state_input=False,
        ),
        data=LeRobotUmiPrechunkedEefRot6dHand30DataConfig(
            repo_id="/mnt/data/dzq/umi_v2/datasets/task_v2_x2_high_lerobot_10hz_h50_right_eye_v2_wrist180",
            assets=AssetsConfig(
                assets_dir="/mnt/data/dzq/openpi/data/assets",
                asset_id="umi_task_v2_x2_high_hand_pose_10hz_h50_right_eye_wrist180_masked_v1",
            ),
            base_config=DataConfig(prompt_from_task=True),
            default_prompt="Put the two objects into the box.",
            use_head_camera=True,
            head_crop_normalized=None,
            head_crop_expected_aspect_ratio=cx002_policy.TASK_V1_HEAD_INPUT_ASPECT_RATIO,
            use_delta_eef_actions=False,
            output_absolute_eef_actions=False,
            sample_filter_key=None,
            action_padding_mask_key="action_is_pad",
            required_action_horizon=50,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/mnt/data/checkpoints/pi05_base/params"
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=500, peak_lr=2.5e-5, decay_steps=20_000, decay_lr=2.5e-6,
        ),
        batch_size=32,
        num_workers=32,
        num_train_steps=10_000,
        log_interval=50,
        save_interval=1_000,
        keep_period=1_000,
        wandb_enabled=False,
        policy_metadata={
            "task_prompt": "Put the two objects into the box.",
            "dataset_variant": "task_v2_x2_high_right_eye_v2_wrist180",
            "selected_source_episodes": 85,
            "excluded_source_episodes": 10,
            "episode_selection_sha256": "10529d85a309173ca01348bb092ca36278d8064737656b88994baa90152f4b2b",
            "original_episode_selection_sha256": "b6a05245e6b0eaf68746fa43851db2f2b097a20f87e0131444f692082a4ebffd",
            "discontinuity_policy": "exclude_10_whole_episodes_no_splitting",
            "dataset_fps": 10,
            "action_horizon": 50,
            "action_horizon_seconds": 5.0,
            "state_input_mode": "none",
            "discrete_state_input": False,
            "state_values_used_for_conditioning": False,
            "state_field_retained_for_interface": True,
            "image_mode": "complete-right-eye-letterbox-no-fixed-roi",
            "head_crop_enabled": False,
            "head_source_actual_wh": [3840, 1200],
            "head_eye_crop_xywh": [1920, 0, 1920, 1200],
            "head_eye_selection": "decoded_width_right_half",
            "head_video_wh": [640, 480],
            "head_input_expected_aspect": "4:3",
            "head_resize": "ResizeImages-224x224-with-pad",
            "training_augmentation": "unchanged pi05 head 95pct random crop, +/-5deg rotation, color jitter",
            "action_padding_mask_key": "action_is_pad",
            "fresh_base_model": True,
            "wrist_rotation_degrees": {"left": 180, "right": 180},
            "training_wrist_rotation_source": "encoded_dataset",
            "additional_training_wrist_rotation": False,
            "inference_wrist_rotation_required": True,
            "paired_state_ablation_config": "pi05_umi_task_v2_x2_high_10hz_h50_wrist180_state_10k_b32_w32_4gpu_v1",
            "comparison_group": "wrist180_state_ablation_4gpu_20260911",
        },
        fsdp_devices=4,
    ),
    TrainConfig(
        name="pi05_umi_task_v2_x2_high_10hz_h50_wrist180_state_10k_b32_w32_4gpu_v1",
        assets_base_dir="/mnt/data/dzq/openpi/data/assets",
        checkpoint_base_dir="/mnt/data/dzq/openpi/checkpoints",
        model=pi0_config.Pi0Config(
            pi05=True, action_dim=32, action_horizon=50, discrete_state_input=True,
        ),
        data=LeRobotUmiPrechunkedEefRot6dHand30DataConfig(
            repo_id="/mnt/data/dzq/umi_v2/datasets/task_v2_x2_high_lerobot_10hz_h50_right_eye_v2_wrist180",
            assets=AssetsConfig(
                assets_dir="/mnt/data/dzq/openpi/data/assets",
                asset_id="umi_task_v2_x2_high_hand_pose_10hz_h50_right_eye_wrist180_masked_v1",
            ),
            base_config=DataConfig(prompt_from_task=True),
            default_prompt="Put the two objects into the box.",
            use_head_camera=True,
            head_crop_normalized=None,
            head_crop_expected_aspect_ratio=cx002_policy.TASK_V1_HEAD_INPUT_ASPECT_RATIO,
            use_delta_eef_actions=False,
            output_absolute_eef_actions=False,
            sample_filter_key=None,
            action_padding_mask_key="action_is_pad",
            required_action_horizon=50,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/mnt/data/checkpoints/pi05_base/params"
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=500, peak_lr=2.5e-5, decay_steps=20_000, decay_lr=2.5e-6,
        ),
        batch_size=32,
        num_workers=32,
        num_train_steps=10_000,
        log_interval=50,
        save_interval=1_000,
        keep_period=1_000,
        wandb_enabled=False,
        policy_metadata={
            "task_prompt": "Put the two objects into the box.",
            "dataset_variant": "task_v2_x2_high_right_eye_v2_wrist180",
            "selected_source_episodes": 85,
            "excluded_source_episodes": 10,
            "episode_selection_sha256": "10529d85a309173ca01348bb092ca36278d8064737656b88994baa90152f4b2b",
            "original_episode_selection_sha256": "b6a05245e6b0eaf68746fa43851db2f2b097a20f87e0131444f692082a4ebffd",
            "discontinuity_policy": "exclude_10_whole_episodes_no_splitting",
            "dataset_fps": 10,
            "action_horizon": 50,
            "action_horizon_seconds": 5.0,
            "state_input_mode": "pi05_discrete_tokens",
            "discrete_state_input": True,
            "state_values_used_for_conditioning": True,
            "state_field_retained_for_interface": True,
            "image_mode": "complete-right-eye-letterbox-no-fixed-roi",
            "head_crop_enabled": False,
            "head_source_actual_wh": [3840, 1200],
            "head_eye_crop_xywh": [1920, 0, 1920, 1200],
            "head_eye_selection": "decoded_width_right_half",
            "head_video_wh": [640, 480],
            "head_input_expected_aspect": "4:3",
            "head_resize": "ResizeImages-224x224-with-pad",
            "training_augmentation": "unchanged pi05 head 95pct random crop, +/-5deg rotation, color jitter",
            "action_padding_mask_key": "action_is_pad",
            "fresh_base_model": True,
            "wrist_rotation_degrees": {"left": 180, "right": 180},
            "training_wrist_rotation_source": "encoded_dataset",
            "additional_training_wrist_rotation": False,
            "inference_wrist_rotation_required": True,
            "paired_state_ablation_config": "pi05_umi_task_v2_x2_high_10hz_h50_wrist180_no_state_10k_b32_w32_4gpu_v1",
            "comparison_group": "wrist180_state_ablation_4gpu_20260911",
        },
        fsdp_devices=4,
    ),
    # task_electric: the explicitly selected 115 sessions, right E6 eye,
    # both wrist videos rotated 180 degrees once at conversion time.
    TrainConfig(
        name="pi05_umi_task_electric_10hz_h50_wrist180_state_20k_b64_w32_8gpu_v1",
        assets_base_dir="/mnt/data/dzq/openpi/data/assets",
        checkpoint_base_dir="/mnt/data/dzq/openpi/checkpoints",
        model=pi0_config.Pi0Config(
            pi05=True, action_dim=32, action_horizon=50, discrete_state_input=True,
        ),
        data=LeRobotUmiPrechunkedEefRot6dHand30DataConfig(
            repo_id="/mnt/data/dzq/umi_v2/datasets/task_electric_lerobot_10hz_h50_right_eye_wrist180_v1",
            assets=AssetsConfig(
                assets_dir="/mnt/data/dzq/openpi/data/assets",
                asset_id="umi_task_electric_10hz_h50_right_eye_wrist180_masked_v1",
            ),
            base_config=DataConfig(prompt_from_task=True),
            default_prompt="Insert the battery into the empty slot in the middle.",
            use_head_camera=True,
            head_crop_normalized=None,
            head_crop_expected_aspect_ratio=cx002_policy.TASK_V1_HEAD_INPUT_ASPECT_RATIO,
            use_delta_eef_actions=False,
            output_absolute_eef_actions=False,
            sample_filter_key=None,
            action_padding_mask_key="action_is_pad",
            required_action_horizon=50,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("/mnt/data/checkpoints/pi05_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=500, peak_lr=2.5e-5, decay_steps=20_000, decay_lr=2.5e-6,
        ),
        batch_size=64,
        num_workers=32,
        num_train_steps=20_000,
        log_interval=50,
        save_interval=1_000,
        keep_period=1_000,
        wandb_enabled=False,
        policy_metadata={
            "task_prompt": "Insert the battery into the empty slot in the middle.",
            "dataset_variant": "task_electric_right_eye_wrist180_v1",
            "selected_source_episodes": 115,
            "excluded_selected_source_episodes": 0,
            "episode_selection_sha256": "0145c1b2005d0a7d93f02e465dba74d156fc9c01474d55ae15712f32b362aa16",
            "discontinuity_policy": "one_source_one_output_no_splitting",
            "dataset_fps": 10,
            "action_horizon": 50,
            "action_horizon_seconds": 5.0,
            "state_input_mode": "pi05_discrete_tokens",
            "discrete_state_input": True,
            "state_values_used_for_conditioning": True,
            "state_field_retained_for_interface": True,
            "image_mode": "complete-right-eye-letterbox-no-fixed-roi",
            "head_crop_enabled": False,
            "head_eye_selection": "decoded_width_right_half",
            "head_video_wh": [640, 480],
            "head_input_expected_aspect": "4:3",
            "head_resize": "ResizeImages-224x224-with-pad",
            "wrist_rotation_degrees": {"left": 180, "right": 180},
            "training_wrist_rotation_source": "encoded_dataset",
            "additional_training_wrist_rotation": False,
            "inference_wrist_rotation_required": True,
            "action_padding_mask_key": "action_is_pad",
            "fresh_base_model": True,
        },
        fsdp_devices=8,
    ),
    # task_v1 30 Hz/H90 dataset. Actions are already prechunked in each row;
    # action_is_pad masks only padded future slots, while every observation
    # anchor remains available for training.
    TrainConfig(
        name="pi05_umi_task_v1_hand_pose_30hz_h90_masked_with_head_roi_v1",
        assets_base_dir="/mnt/data/dzq/openpi/data/assets",
        checkpoint_base_dir="/mnt/data/dzq/openpi/checkpoints",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=90,
            discrete_state_input=True,
        ),
        data=LeRobotUmiPrechunkedEefRot6dHand30DataConfig(
            repo_id="/mnt/data/dzq/umi/datasets/task_v1_lerobot_30hz_h90",
            assets=AssetsConfig(
                assets_dir="/mnt/data/dzq/openpi/data/assets",
                asset_id="umi_task_v1_hand_pose_30hz_h90_masked_v1",
            ),
            base_config=DataConfig(prompt_from_task=True),
            default_prompt=_UMI_TASK_V1_PROMPT,
            use_head_camera=True,
            head_crop_normalized=cx002_policy.TASK_V1_HEAD_CROP_NORMALIZED,
            head_crop_expected_aspect_ratio=cx002_policy.TASK_V1_HEAD_INPUT_ASPECT_RATIO,
            use_delta_eef_actions=False,
            output_absolute_eef_actions=False,
            sample_filter_key=None,
            action_padding_mask_key="action_is_pad",
            required_action_horizon=90,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/mnt/data/checkpoints/pi05_base/params"
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=500,
            peak_lr=2.5e-5,
            decay_steps=20_000,
            decay_lr=2.5e-6,
        ),
        batch_size=64,
        num_workers=6,
        num_train_steps=20_000,
        log_interval=50,
        save_interval=2_000,
        keep_period=2_000,
        wandb_enabled=False,
        policy_metadata={
            "task_prompt": _UMI_TASK_V1_PROMPT,
            "dataset_fps": 30,
            "action_horizon": 90,
            "action_horizon_seconds": 3.0,
            "image_mode": "training-roi-cropped",
            "head_crop_normalized_ltrb": cx002_policy.TASK_V1_HEAD_CROP_NORMALIZED,
            "head_crop_coordinate_order": "left,top,right,bottom",
            "head_crop_expected_input_aspect": "4:3",
            "head_crop_applied_by": "CX002Inputs-before-ResizeImages",
            "action_padding_mask_key": "action_is_pad",
        },
        fsdp_devices=8,
    ),
    # Deployment-only companion for temporarily serving an ROI-trained
    # checkpoint with the complete 4:3 head frame. This intentionally does not
    # match the training visual distribution and must not be used as a training
    # config or treated as the final evaluation setup.
    TrainConfig(
        name="pi05_umi_task_v1_hand_pose_10hz_h50_masked_with_head_roi_v1_deploy_uncropped",
        assets_base_dir="/mnt/data/dzq/openpi/data/assets",
        checkpoint_base_dir="/mnt/data/dzq/openpi/checkpoints",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            discrete_state_input=True,
        ),
        data=LeRobotUmiPrechunkedEefRot6dHand30DataConfig(
            repo_id="/mnt/data/dzq/umi/datasets/task_v1_lerobot_h50",
            assets=AssetsConfig(
                assets_dir="/mnt/data/dzq/openpi/data/assets",
                asset_id="umi_task_v1_hand_pose_10hz_h50_masked_v1",
            ),
            base_config=DataConfig(prompt_from_task=True),
            default_prompt=_UMI_TASK_V1_PROMPT,
            use_head_camera=True,
            head_crop_normalized=None,
            head_crop_expected_aspect_ratio=cx002_policy.TASK_V1_HEAD_INPUT_ASPECT_RATIO,
            use_delta_eef_actions=False,
            output_absolute_eef_actions=False,
            sample_filter_key=None,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/mnt/data/checkpoints/pi05_base/params"
        ),
        batch_size=1,
        num_workers=0,
        num_train_steps=0,
        wandb_enabled=False,
        policy_metadata={
            "task_prompt": _UMI_TASK_V1_PROMPT,
            "image_mode": "deployment-full-frame-uncropped",
            "head_crop_enabled": False,
            "head_input_expected_aspect": "4:3",
            "checkpoint_training_image_mode": "training-roi-cropped",
            "distribution_mismatch_warning": (
                "This temporary deployment mode uses a full head frame for a checkpoint "
                "trained on the ROI crop."
            ),
        },
        fsdp_devices=1,
    ),
    # taskumi2 hand-pose H50, using E6 head RGB plus both wrist cameras.
    TrainConfig(
        name="pi05_umi_taskumi2_hand_pose_10hz_h50_with_head_v1",
        assets_base_dir="/mnt/data/dzq/openpi/data/assets",
        checkpoint_base_dir="/mnt/data/dzq/openpi/checkpoints",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            discrete_state_input=True,
        ),
        data=LeRobotUmiPrechunkedEefRot6dHand30DataConfig(
            repo_id="/mnt/data/dzq/umi/datasets/taskumi2",
            assets=AssetsConfig(
                assets_dir="/mnt/data/dzq/openpi/data/assets",
                asset_id="umi_taskumi2_hand_pose_10hz_h50_v1",
            ),
            base_config=DataConfig(prompt_from_task=True),
            default_prompt=_UMI_BOX_IN_OUT_PROMPT,
            use_head_camera=True,
            use_delta_eef_actions=False,
            output_absolute_eef_actions=False,
            sample_filter_key="sample_valid_h50",
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/mnt/data/checkpoints/pi05_base/params"
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=500,
            peak_lr=2.5e-5,
            decay_steps=20_000,
            decay_lr=2.5e-6,
        ),
        batch_size=64,
        num_workers=6,
        num_train_steps=20_000,
        log_interval=50,
        save_interval=2_000,
        keep_period=2_000,
        wandb_enabled=False,
        fsdp_devices=8,
    ),
    # taskumi3 hand-pose-v2 H50, using E6 head RGB plus both wrist cameras.
    TrainConfig(
        name="pi05_umi_taskumi3_hand_pose_v2_10hz_h50_with_head_v1",
        assets_base_dir="/mnt/data/dzq/openpi/data/assets",
        checkpoint_base_dir="/mnt/data/dzq/openpi/checkpoints",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            discrete_state_input=True,
        ),
        data=LeRobotUmiPrechunkedEefRot6dHand30DataConfig(
            repo_id="/mnt/data/dzq/umi/datasets/taskumi3",
            assets=AssetsConfig(
                assets_dir="/mnt/data/dzq/openpi/data/assets",
                asset_id="umi_taskumi3_hand_pose_v2_10hz_h50_v1",
            ),
            base_config=DataConfig(prompt_from_task=True),
            default_prompt=_UMI_BOX_IN_OUT_PROMPT,
            use_head_camera=True,
            use_delta_eef_actions=False,
            output_absolute_eef_actions=False,
            sample_filter_key="sample_valid_h50",
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/mnt/data/checkpoints/pi05_base/params"
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=500,
            peak_lr=2.5e-5,
            decay_steps=20_000,
            decay_lr=2.5e-6,
        ),
        batch_size=64,
        num_workers=6,
        num_train_steps=20_000,
        log_interval=50,
        save_interval=2_000,
        keep_period=2_000,
        wandb_enabled=False,
        fsdp_devices=8,
    ),
    # taskumi3 hand-pose-v2 H50, using only the two wrist cameras.  The fixed
    # head slot is a masked black image so the pi0.5 image layout stays valid.
    TrainConfig(
        name="pi05_umi_taskumi3_hand_pose_v2_10hz_h50_wrist_only_v1",
        assets_base_dir="/mnt/data/dzq/openpi/data/assets",
        checkpoint_base_dir="/mnt/data/dzq/openpi/checkpoints",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            discrete_state_input=True,
        ),
        data=LeRobotUmiPrechunkedEefRot6dHand30DataConfig(
            repo_id="/mnt/data/dzq/umi/datasets/taskumi3",
            assets=AssetsConfig(
                assets_dir="/mnt/data/dzq/openpi/data/assets",
                asset_id="umi_taskumi3_hand_pose_v2_10hz_h50_v1",
            ),
            base_config=DataConfig(prompt_from_task=True),
            default_prompt=_UMI_BOX_IN_OUT_PROMPT,
            use_head_camera=False,
            use_delta_eef_actions=False,
            output_absolute_eef_actions=False,
            sample_filter_key="sample_valid_h50",
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/mnt/data/checkpoints/pi05_base/params"
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=500,
            peak_lr=2.5e-5,
            decay_steps=20_000,
            decay_lr=2.5e-6,
        ),
        batch_size=64,
        num_workers=6,
        num_train_steps=20_000,
        log_interval=50,
        save_interval=2_000,
        keep_period=2_000,
        wandb_enabled=False,
        fsdp_devices=8,
    ),
    # taskumi2 hand-pose H50, using only the two wrist cameras.  The fixed
    # head slot is a masked black image so the pi0.5 image layout stays valid.
    TrainConfig(
        name="pi05_umi_taskumi2_hand_pose_10hz_h50_wrist_only_v1",
        assets_base_dir="/mnt/data/dzq/openpi/data/assets",
        checkpoint_base_dir="/mnt/data/dzq/openpi/checkpoints",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            discrete_state_input=True,
        ),
        data=LeRobotUmiPrechunkedEefRot6dHand30DataConfig(
            repo_id="/mnt/data/dzq/umi/datasets/taskumi2",
            assets=AssetsConfig(
                assets_dir="/mnt/data/dzq/openpi/data/assets",
                asset_id="umi_taskumi2_hand_pose_10hz_h50_v1",
            ),
            base_config=DataConfig(prompt_from_task=True),
            default_prompt=_UMI_BOX_IN_OUT_PROMPT,
            use_head_camera=False,
            use_delta_eef_actions=False,
            output_absolute_eef_actions=False,
            sample_filter_key="sample_valid_h50",
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/mnt/data/checkpoints/pi05_base/params"
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=500,
            peak_lr=2.5e-5,
            decay_steps=20_000,
            decay_lr=2.5e-6,
        ),
        batch_size=64,
        num_workers=6,
        num_train_steps=20_000,
        log_interval=50,
        save_interval=2_000,
        keep_period=2_000,
        wandb_enabled=False,
        fsdp_devices=8,
    ),
    TrainConfig(
        name="pi05_umi_latest100_wrist_only_spatial_eef_hand24",

        assets_base_dir="/mnt/data/dzq/openpi/data/assets",
        checkpoint_base_dir="/mnt/data/dzq/openpi/checkpoints",

        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=30,
            discrete_state_input=True,
        ),

        data=LeRobotUmiSpatialEefHand24DataConfig(
            # 可以继续使用现有三视角数据集；
            # CX002Inputs 会将头部相机替换为黑图并设置 mask=False。
            repo_id=(
                "/mnt/data/dzq/umi/datasets/"
                "latest100_lerobot_three_view_eef12_hand12"
            ),

            # 使用独立 asset，避免与三相机实验混淆。
            assets=AssetsConfig(
                asset_id="latest100_wrist_only_spatial_eef_hand24"
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
            default_prompt=_UMI_LATEST100_PROMPT,

            # 仅使用左右腕部相机。
            use_head_camera=False,

            # 保持原来的动作表示。
            use_delta_eef_actions=False,
            output_absolute_eef_actions=False,
            sample_filter_key="sample_valid_h30",
        ),

        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/mnt/data/checkpoints/pi05_base/params"
        ),

        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=500,
            peak_lr=2.5e-5,
            decay_steps=20_000,
            decay_lr=2.5e-6,
        ),
        # Eight A100s, eight samples per device.
        batch_size=64,
        num_workers=6,
        num_train_steps=20_000,

        log_interval=50,
        save_interval=2_000,
        keep_period=2_000,

        wandb_enabled=False,
        fsdp_devices=1,
    ),
    # Latest-100 UMI three-view fine-tuning.  Controller poses are aligned to
    # robot X-forward/Y-left/Z-up and act as dual EEF proxies. State is
    # absolute EEF12 + absolute hand12, so pi0.5 must tokenize it. H30 EEF
    # actions become spatial shared-anchor deltas while hand12 stays absolute.
    TrainConfig(
        name="pi05_umi_latest100_spatial_eef_hand24",
        assets_base_dir="/mnt/data/dzq/openpi/data/assets",
        checkpoint_base_dir="/mnt/data/dzq/openpi/checkpoints",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=30,
            discrete_state_input=True,
        ),
        data=LeRobotUmiSpatialEefHand24DataConfig(
            repo_id="/mnt/data/dzq/umi/data/latest100_lerobot_three_view_eef12_hand12",
            assets=AssetsConfig(asset_id="latest100_three_view_spatial_eef_hand24"),
            base_config=DataConfig(prompt_from_task=True),
            default_prompt=_UMI_LATEST100_PROMPT,
            use_delta_eef_actions=False,
            output_absolute_eef_actions=False,
            sample_filter_key="sample_valid_h30",
            
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/mnt/data/checkpoints/pi05_base/params"
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=500,
            peak_lr=2.5e-5,
            decay_steps=20_000,
            decay_lr=2.5e-6,
        ),
        # Eight A100s, eight samples per device.
        batch_size=64,
        num_workers=6,
        num_train_steps=20_000,
        log_interval=50,
        save_interval=2_000,
        keep_period=2_000,
        wandb_enabled=False,
        fsdp_devices=1,
    ),
    #
    # Task2 stores raw14 next-frame absolute dual-EEF xyz+rpy targets:
    # left EEF6, right EEF6, left gripper, right gripper. The data config
    # automatically converts every 30-step future target to a delta anchored
    # at the current state before normalization and JAX training.
    #
    TrainConfig(
        name="pi05_cx002_task2_eef6",
        assets_base_dir="/mnt/data/dzq/openpi/data/assets",
        checkpoint_base_dir="/mnt/data/dzq/openpi/checkpoints",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=30,
            discrete_state_input=False,
        ),
        data=LeRobotCX002RawEef6DataConfig(
            repo_id="/mnt/data/dzq/openpi/datasets/task2_eef6",
            assets=AssetsConfig(asset_id="task2_eef6"),
            base_config=DataConfig(prompt_from_task=True),
            default_prompt=_CX002_TASK2_PROMPT,
            use_delta_eef_actions=True,
            output_absolute_eef_actions=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/mnt/data/checkpoints/pi05_base/params"
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=500,
            peak_lr=2.5e-5,
            decay_steps=20_000,
            decay_lr=2.5e-6,
        ),
        batch_size=64,
        num_workers=8,
        num_train_steps=20_000,
        log_interval=50,
        save_interval=2_000,
        keep_period=2_000,
        wandb_enabled=False,
        fsdp_devices=1,
    ),
    #自己进行修改需要的数据
    TrainConfig(
        name = "pi05_cx002_task4",
        assets_base_dir="/mnt/data/dzq/openpi/data/assets",
        checkpoint_base_dir="/mnt/data/dzq/openpi/checkpoints",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=30,
            discrete_state_input=False,
        ),
        data=LeRobotCX002RawEef6DataConfig(
            repo_id="/mnt/data/dzq/openpi/datasets/task4",
            assets=AssetsConfig(asset_id="task4"),
            base_config=DataConfig(prompt_from_task=True),
            default_prompt=_CX002_TASK4_PROMPT,
            use_delta_eef_actions=True,
            output_absolute_eef_actions=True,
        ),
                weight_loader=weight_loaders.CheckpointWeightLoader(
            "/mnt/data/checkpoints/pi05_base/params"
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=500,
            peak_lr=2.5e-5,
            decay_steps=10_000,
            decay_lr=2.5e-6,
        ),
        batch_size=56,
        num_workers=8,
        num_train_steps=10_000,
        log_interval=50,
        save_interval=2_000,
        keep_period=2_000,
        wandb_enabled=False,
        fsdp_devices=1,
    ),
    TrainConfig(
        name="pi05_cx002_task5",
        assets_base_dir="/mnt/data/dzq/openpi/data/assets",
        checkpoint_base_dir="/mnt/data/dzq/openpi/checkpoints",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            discrete_state_input=False,
        ),
        data=LeRobotCX002RawEef6DataConfig(
            repo_id="/mnt/data/dzq/openpi/datasets/task5",
            assets=AssetsConfig(asset_id="task5"),
            base_config=DataConfig(prompt_from_task=True),
            default_prompt=_CX002_TASK5_PROMPT,
            use_delta_eef_actions=True,
            output_absolute_eef_actions=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/mnt/data/checkpoints/pi05_base/params"
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=500,
            peak_lr=2.5e-5,
            decay_steps=20_000,
            decay_lr=2.5e-6,
        ),
        # Seven training GPUs are used (physical 0,2-7). Keep the global
        # batch divisible by seven and leave physical GPU 1 to its owner.
        batch_size=42,
        num_workers=6,
        num_train_steps=20_000,
        log_interval=50,
        save_interval=2_000,
        keep_period=2_000,
        wandb_enabled=False,
        fsdp_devices=1,
    ),
    TrainConfig(
        name = "pi05_cx002_task1",
        assets_base_dir="/mnt/data/dzq/openpi/data/assets",
        checkpoint_base_dir="/mnt/data/dzq/openpi/checkpoints",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=75,
            discrete_state_input=False,
        ),
        data=LeRobotCX002RawEef6DataConfig(
            repo_id="/mnt/data/dzq/openpi/datasets/task1",
            assets=AssetsConfig(asset_id="task1"),
            base_config=DataConfig(prompt_from_task=True),
            default_prompt=_CX002_TASK1_PROMPT,
            use_delta_eef_actions=True,
            output_absolute_eef_actions=True,
        ),
            weight_loader=weight_loaders.CheckpointWeightLoader(
            "/mnt/data/checkpoints/pi05_base/params"
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=500,
            peak_lr=2.5e-5,
            decay_steps=20_000,
            decay_lr=2.5e-6,
        ),
        batch_size=48,
        num_workers=8,
        num_train_steps=20_000,
        log_interval=50,
        save_interval=2_000,
        keep_period=2_000,
        wandb_enabled=False,
        fsdp_devices=1,
    ),
    TrainConfig(
        name = "pi05_cx002_task3",
        assets_base_dir="/mnt/data/dzq/openpi/data/assets",
        checkpoint_base_dir="/mnt/data/dzq/openpi/checkpoints",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            discrete_state_input=False,
        ),
        data=LeRobotCX002RawEef6DataConfig(
            repo_id="/mnt/data/dzq/openpi/datasets/task3",
            assets=AssetsConfig(asset_id="task3"),
            base_config=DataConfig(prompt_from_task=True),
            default_prompt=_CX002_TASK3_PROMPT,
            use_delta_eef_actions=True,
            output_absolute_eef_actions=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/mnt/data/checkpoints/pi05_base/params"
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=500,
            peak_lr=2.5e-5,
            decay_steps=20_000,
            decay_lr=2.5e-6,
        ),
        batch_size=48,
        num_workers=6,
        num_train_steps=20_000,
        log_interval=50,
        save_interval=2_000,
        keep_period=2_000,
        wandb_enabled=True,
        fsdp_devices=1,
    ),

    #
    # Fine-tuning Aloha configs.
    #
    # This is a test config that is used to illustate how train on a custom LeRobot dataset.
    # For instructions on how to convert and train on your own Aloha dataset see examples/aloha_real/README.md
    TrainConfig(
        name="pi0_aloha_pen_uncap",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            repo_id="physical-intelligence/aloha_pen_uncap_diverse",
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
                asset_id="trossen",
            ),
            default_prompt="uncap the pen",
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
    ),
    TrainConfig(
        name="pi05_aloha_pen_uncap",
        model=pi0_config.Pi0Config(pi05=True),
        data=LeRobotAlohaDataConfig(
            repo_id="physical-intelligence/aloha_pen_uncap_diverse",
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi05_base/assets",
                asset_id="trossen",
            ),
            default_prompt="uncap the pen",
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=20_000,
        batch_size=64,
    ),
    #
    # Fine-tuning DROID configs.
    #
    TrainConfig(
        # This config is for fine-tuning pi0-FAST-base on the *full* DROID dataset.
        # We use RLDS data loading to make training on this large dataset tractable.
        # For fine-tuning on your own DROID dataset, see below.
        name="pi0_fast_full_droid_finetune",
        model=pi0_fast.Pi0FASTConfig(
            action_dim=8,
            action_horizon=16,
            max_token_len=180,
        ),
        data=RLDSDroidDataConfig(
            repo_id="droid",
            # Set this to the path to your DROID RLDS dataset (the parent directory of the `droid` directory).
            rlds_data_dir="<path_to_droid_rlds_dataset>",
            action_space=droid_rlds_dataset.DroidActionSpace.JOINT_POSITION,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        num_train_steps=100_000,  # 100k steps should be sufficient, takes ~2 days on 8x H100s
        batch_size=256,
        log_interval=100,
        save_interval=5000,
        keep_period=20_000,
        num_workers=0,  # Important: RLDS DataLoader requires num_workers=0, handles multi-processing internally
    ),
    TrainConfig(
        # This config is for fine-tuning pi05 on the *full* DROID dataset.
        # We use RLDS data loading to make training on this large dataset tractable.
        # For fine-tuning on your own DROID dataset, see below.
        name="pi05_full_droid_finetune",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
        ),
        data=RLDSDroidDataConfig(
            repo_id="droid",
            # Set this to the path to your DROID RLDS dataset (the parent directory of the `droid` directory).
            rlds_data_dir="/mnt/pi-data/kevin",
            action_space=droid_rlds_dataset.DroidActionSpace.JOINT_POSITION,
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi05_base/assets/",
                asset_id="droid",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        num_train_steps=100_000,
        batch_size=256,
        log_interval=100,
        save_interval=5000,
        keep_period=10_000,
        num_workers=0,  # Important: RLDS DataLoader requires num_workers=0, handles multi-processing internally
    ),
    TrainConfig(
        # This config is for fine-tuning pi05-DROID on a custom (smaller) DROID dataset.
        # Here, we use LeRobot data format (like for all other fine-tuning examples)
        # To convert your custom DROID dataset (<10s of hours) to LeRobot format, see examples/droid/convert_droid_data_to_lerobot.py
        name="pi05_droid_finetune",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,  # pi05 is trained with 32-dim actions
            action_horizon=16,
        ),
        data=LeRobotDROIDDataConfig(
            # Replace with your custom DROID LeRobot dataset repo id.
            repo_id="your_hf_username/my_droid_dataset",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(
                # Important: reuse the original DROID norm stats during fine-tuning!
                assets_dir="gs://openpi-assets/checkpoints/pi05_droid/assets",
                asset_id="droid",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_droid/params"),
        num_train_steps=20_000,
        batch_size=32,
    ),
    #
    # ALOHA Sim configs. This config is used to demonstrate how to train on a simple simulated environment.
    #
    TrainConfig(
        name="pi0_aloha_sim",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            repo_id="lerobot/aloha_sim_transfer_cube_human",
            default_prompt="Transfer cube",
            use_delta_joint_actions=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
    ),
    #
    # Debugging configs.
    #
    TrainConfig(
        name="debug",
        data=FakeDataConfig(),
        batch_size=2,
        model=pi0_config.Pi0Config(paligemma_variant="dummy", action_expert_variant="dummy"),
        save_interval=100,
        overwrite=True,
        exp_name="debug",
        num_train_steps=10,
        wandb_enabled=False,
    ),
    TrainConfig(
        name="debug_restore",
        data=FakeDataConfig(),
        batch_size=2,
        model=pi0_config.Pi0Config(paligemma_variant="dummy", action_expert_variant="dummy"),
        weight_loader=weight_loaders.CheckpointWeightLoader("./checkpoints/debug/debug/9/params"),
        overwrite=True,
        exp_name="debug",
        num_train_steps=10,
        wandb_enabled=False,
    ),
    TrainConfig(
        name="debug_pi05",
        model=pi0_config.Pi0Config(pi05=True, paligemma_variant="dummy", action_expert_variant="dummy"),
        data=FakeDataConfig(),
        batch_size=2,
        num_train_steps=10,
        overwrite=True,
        exp_name="debug_pi05",
        wandb_enabled=False,
    ),
    # RoboArena & PolaRiS configs.
    *roboarena_config.get_roboarena_configs(),
    *polaris_config.get_polaris_configs(),
    # taskumi4 hand-pose-v3 H50, using E6 head RGB plus both wrist cameras.
    TrainConfig(
        name="pi05_umi_taskumi4_hand_pose_v3_10hz_h50_with_head_v1",
        assets_base_dir="/mnt/data/dzq/openpi/data/assets",
        checkpoint_base_dir="/mnt/data/dzq/openpi/checkpoints",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            discrete_state_input=True,
        ),
        data=LeRobotUmiPrechunkedEefRot6dHand30DataConfig(
            repo_id="/mnt/data/dzq/umi/datasets/taskumi4",
            assets=AssetsConfig(
                assets_dir="/mnt/data/dzq/openpi/data/assets",
                asset_id="umi_taskumi4_hand_pose_v3_10hz_h50_v1",
            ),
            base_config=DataConfig(prompt_from_task=True),
            default_prompt=_UMI_BOX_IN_OUT_PROMPT,
            use_head_camera=True,
            use_delta_eef_actions=False,
            output_absolute_eef_actions=False,
            sample_filter_key="sample_valid_h50",
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/mnt/data/checkpoints/pi05_base/params"
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=500,
            peak_lr=2.5e-5,
            decay_steps=10_000,
            decay_lr=2.5e-6,
        ),
        batch_size=42,
        num_workers=6,
        num_train_steps=10_000,
        log_interval=50,
        save_interval=2_000,
        keep_period=2_000,
        wandb_enabled=False,
        fsdp_devices=1,
    ),



]

# User-requested task_electric restart: change loader workers only. Keep the
# original 32-worker recipe and its historical experiment fully intact.
_CONFIGS.append(dataclasses.replace(
    next(config for config in _CONFIGS
         if config.name == "pi05_umi_task_electric_10hz_h50_wrist180_state_20k_b64_w32_8gpu_v1"),
    name="pi05_umi_task_electric_10hz_h50_wrist180_state_20k_b64_w128_8gpu_v1",
    num_workers=128,
))

if len({config.name for config in _CONFIGS}) != len(_CONFIGS):
    raise ValueError("Config names must be unique.")
_CONFIGS_DICT = {config.name: config for config in _CONFIGS}


def cli() -> TrainConfig:
    return tyro.extras.overridable_config_cli({k: (k, v) for k, v in _CONFIGS_DICT.items()})


def get_config(config_name: str) -> TrainConfig:
    """Get a config by name."""
    if config_name not in _CONFIGS_DICT:
        closest = difflib.get_close_matches(config_name, _CONFIGS_DICT.keys(), n=1, cutoff=0.0)
        closest_str = f" Did you mean '{closest[0]}'? " if closest else ""
        raise ValueError(f"Config '{config_name}' not found.{closest_str}")

    return _CONFIGS_DICT[config_name]
