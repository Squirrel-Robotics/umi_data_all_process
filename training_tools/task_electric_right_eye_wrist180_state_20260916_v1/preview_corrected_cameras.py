"""Render the first aligned raw observation with the corrected video-only mapping."""
import io
import subprocess
from PIL import Image,ImageDraw,ImageFont
from experiment import *
import converter_electric as converter

def main():
    episode=(ROOT/'pass_sessions.txt').read_text().splitlines()[0]
    plan=converter.make_plan(converter.locate_episode_files(SOURCE/episode),0,10,50,100000000,100000000,'nearest')
    head=converter.probe_e6_right_eye(plan.files.head_video)
    roles={'head':(plan.files.head_video,plan.e6_source_row_indices,'hevc',head['source_crop']),
           'cam0':(plan.files.cam0_video,plan.cam0_frame_indices,'mjpeg',None),
           'cam1':(plan.files.cam1_video,plan.cam1_frame_indices,'mjpeg',None)}
    labels={'head':'HEAD: E6 right eye','cam1':'LEFT wrist: cam1 / 180 deg','cam0':'RIGHT wrist: cam0 / 180 deg'}
    output=ROOT/'camera_mapping_preview';output.mkdir(exist_ok=False)
    canvas=Image.new('RGB',(1920,560),'#171c26');draw=ImageDraw.Draw(canvas)
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',28)
    records=[]
    for column,(key,role) in enumerate(converter.VIDEO_KEYS.items()):
        source,indices,fmt,crop=roles[role];index=int(indices[0])
        filters=[f'select=eq(n\\,{index})']
        if crop:filters.append(crop)
        filters.append(converter.resize_filter(640,480,'letterbox'))
        if role!='head':filters.extend(['hflip','vflip'])
        command=['ffmpeg','-v','error','-threads','2','-f',fmt,'-i',str(source),'-an','-filter_threads','1',
                 '-vf',','.join(filters),'-frames:v','1','-pix_fmt','rgb24','-f','image2pipe','-vcodec','png','pipe:1']
        data=subprocess.check_output(command,timeout=60)
        image=Image.open(io.BytesIO(data)).convert('RGB')
        image.save(output/(role+'.png'))
        canvas.paste(image,(640*column,40));draw.text((640*column+12,3),labels[role],fill='white',font=font)
        records.append({'key':key,'source':str(source),'source_frame_index':index,'filter':filters})
    draw.text((12,526),f'{episode} | aligned first output frame',fill='white',font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',22))
    canvas.save(output/'corrected_three_cameras.png')
    with (output/'preview.json').open('x') as f:json.dump(records,f,indent=2)
    print(str(output/'corrected_three_cameras.png'),flush=True)

if __name__=='__main__':main()
