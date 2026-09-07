#!/usr/bin/env python3
"""Generate geometric SVGs for the calibration docs, using only stdlib.

Run from repository root: python3 docs/tools/render_calibration_visuals.py
Optional --animation-frames DIR emits SVG frames (render to GIF separately).
All trajectories are synthetic. Body geometry is schematic, not product CAD.
"""
import argparse
import math
from pathlib import Path
from html import escape

XCOL, YCOL, ZCOL = '#f45462', '#43c78d', '#5599ff'
O = [0.007435, -0.015192, -0.053839]
F = [0.007435, -0.100782, -0.109321]
U = [0.007435, -0.020632, -0.045448]
def add(a,b): return [x+y for x,y in zip(a,b)]
def sub(a,b): return [x-y for x,y in zip(a,b)]
def mul(a,s): return [v*s for v in a]
def dot(a,b): return sum(x*y for x,y in zip(a,b))
def cross(a,b): return [a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]]
def unit(a): return mul(a,1/math.sqrt(dot(a,a)))
def trans(a): return [list(r) for r in zip(*a)]
def mv(a,v): return [dot(r,v) for r in a]
def mm(a,b): return [[dot(r,c) for c in trans(b)] for r in a]
I = [[1,0,0],[0,1,0],[0,0,1]]
x=unit(sub(F,O)); y=unit(cross(sub(U,O),x)); z=unit(cross(x,y))
RCH=trans([x,y,z]); RHC=trans(RCH)
CH=mul(mv(RHC,O),-1)
FH=mv(RHC,sub(F,O)); UH=mv(RHC,sub(U,O))
assert abs(dot(x,cross(y,z))-1)<1e-12
assert max(abs(dot(a,b)) for a,b in [(x,y),(y,z),(z,x)])<1e-12

class SVG:
    def __init__(self,title,dark=False):
        self.dark=dark; self.ink='#edf4ff' if dark else '#142b45'
        self.muted='#adbed4' if dark else '#607189'
        bg='#101c30' if dark else '#f5f8fc'
        self.items=[f'<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="940" viewBox="0 0 1600 940" role="img"><title>{escape(title)}</title><rect width="1600" height="940" fill="{bg}"/><style>text{{font-family:"Noto Sans CJK SC","Microsoft YaHei","PingFang SC",sans-serif}}</style>']
        self.text(48,65,title,36)
    def text(self,x,y,s,size=24,color=None):
        self.items.append(f'<text x="{x:.2f}" y="{y:.2f}" font-size="{size}" fill="{color or self.ink}">{escape(s)}</text>')
    def line(self,a,b,color,width=3,dash=False,opacity=1):
        d=' stroke-dasharray="8 7"' if dash else ''
        self.items.append(f'<path d="M{a[0]:.2f},{a[1]:.2f} L{b[0]:.2f},{b[1]:.2f}" fill="none" stroke="{color}" stroke-width="{width}" stroke-linecap="round" opacity="{opacity}"{d}/>')
    def poly(self,points,fill,stroke=None,opacity=1):
        coords=' '.join(f'{a:.2f},{b:.2f}' for a,b in points)
        self.items.append(f'<polygon points="{coords}" fill="{fill}" stroke="{stroke or fill}" stroke-width="1.4" opacity="{opacity}"/>')
    def circle(self,p,r,color,stroke='white'):
        self.items.append(f'<circle cx="{p[0]:.2f}" cy="{p[1]:.2f}" r="{r}" fill="{color}" stroke="{stroke}" stroke-width="2"/>')
    def arrow(self,a,b,color,width=7,opacity=1):
        self.line(a,b,color,width,opacity=opacity)
        d=unit([b[0]-a[0],b[1]-a[1]]); n=[-d[1],d[0]]
        self.poly([b,add(sub(b,mul(d,20)),mul(n,8)),sub(sub(b,mul(d,20)),mul(n,8))],color,opacity=opacity)
    def save(self,path):
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text('\n'.join(self.items+['</svg>'])+'\n')

class Scene:
    def __init__(self,svg,center,scale,R=I,p=(0,0,0)):
        self.s=svg; self.center=center; self.scale=scale; self.R=R; self.p=p
    def world(self,v): return add(mv(self.R,v),self.p)
    def point(self,v):
        a=self.world(v)
        return [self.center[0]+self.scale*dot([.8,-.6,0],a),self.center[1]-self.scale*dot([.3,.4,.8660254],a)]
    def segment(self,a,b,color,width=3,dash=False): self.s.line(self.point(a),self.point(b),color,width,dash)
    def axes(self,o,R=I,length=.055,labels=True,width=7,opacity=1):
        for vec,col,name in zip(trans(R),[XCOL,YCOL,ZCOL],['X','Y','Z']):
            end=self.point(add(o,mul(vec,length)))
            self.s.arrow(self.point(o),end,col,width,opacity)
            if labels: self.s.text(end[0]+8,end[1]-9,name,27,col)
    def body(self,opacity=.7):
        boxes=[([.035,0,-.01],[.065,.060,.02],False)]
        for j,length in enumerate([.040,.052,.048,.034]):
            boxes.append(([.066+length/2,-.024+j*.016,-.01],[length,.011,.017],False))
        boxes.append(([.035,-.040,-.014],[.040,.016,.020],False))
        boxes.append((CH,[.029,.024,.043],True))
        faces=[]
        for origin,size,controller in boxes:
            corners=[add(origin,[sx*size[0]/2,sy*size[1]/2,sz*size[2]/2]) for sx,sy,sz in [(-1,-1,-1),(-1,-1,1),(-1,1,-1),(-1,1,1),(1,-1,-1),(1,-1,1),(1,1,-1),(1,1,1)]]
            colors=['#5784ac','#729abd','#9bb9d2'] if controller else ['#9bacbd','#bdcbd9','#dae3ed']
            for k,inds in enumerate([[1,3,7,5],[0,1,5,4],[0,2,3,1],[4,5,7,6],[2,6,7,3],[0,4,6,2]]):
                verts=[corners[i] for i in inds]
                depth=sum(dot([-.5196,-.6928,.5],self.world(v)) for v in verts)/4
                faces.append((depth,verts,colors[k%3]))
        for _,verts,col in sorted(faces): self.s.poly([self.point(v) for v in verts],col,'#647e98',opacity)
        self.segment(CH,[0,0,0],'#7990a8',5)
    def grid(self):
        for i in range(-5,8):
            a=i*.025
            self.segment([-.10,a,-.045],[.19,a,-.045],'#273b55',1)
            self.segment([a,-.125,-.045],[a,.175,-.045],'#273b55',1)

def point_label(scene,p,label,anchor,color):
    q=scene.point(p); scene.s.circle(q,7,color)
    scene.s.line(q,anchor,scene.s.muted,2)
    scene.s.text(anchor[0]+8,anchor[1]-9,label,25,color)

def calibration(out):
    s=SVG('用三个点，把“手柄坐标系”变成“自己的手部坐标系”')
    s.text(48,107,'测点的位置来自同一个 Controller 局部坐标系；手的外形仅作示意。',23,s.muted)
    s.line([800,154],[800,842],'#d8e2ed',2)
    s.text(48,180,'① 先选 O、F、U',31)
    s.text(848,180,'② 再长出自己的 XYZ',31)
    left=Scene(s,[325,567],3750)
    left.body(.62)
    left.segment([0,0,0],FH,XCOL,5)
    left.segment([0,0,0],UH,'#b88922',5)
    point_label(left,FH,'F：前向点',[475,355],XCOL)
    point_label(left,UH,'U：上向点',[165,372],'#a47b20')
    point_label(left,[0,0,0],'O：手部原点',[150,674],s.ink)
    point_label(left,CH,'Controller',[65,465],'#3b78ad')
    s.text(438,612,'OF ≈ 102 mm',25,XCOL)
    s.text(78,735,'OU ≈ 10 mm',25,'#a47b20')
    right=Scene(s,[1115,590],3750)
    right.body(.35)
    right.axes([0,0,0],length=.084)
    s.circle(right.point([0,0,0]),8,s.ink)
    s.text(1200,405,'X：沿 O → F',25,XCOL)
    s.text(847,449,'Y：叉积确定',25,YCOL)
    s.text(1158,268,'Z：上向正交化',25,ZCOL)
    s.text(930,738,'三根轴互相垂直，随手一起转动',25)
    s.text(48,839,'三个点不是三个轴：O 是原点；F 定前向；U 定上向提示。',27)
    s.text(48,891,'红 X  /  绿 Y  /  蓝 Z      ·      本图使用新硬件右侧测量点；示意外形并非产品 CAD。',22,s.muted)
    s.save(out/'three-points-3d.svg')

def pose(t):
    a=.45*math.sin(2*math.pi*t); b=.32*math.sin(2*math.pi*t+.5)
    rz=[[math.cos(a),-math.sin(a),0],[math.sin(a),math.cos(a),0],[0,0,1]]
    ry=[[math.cos(b),0,math.sin(b)],[0,1,0],[-math.sin(b),0,math.cos(b)]]
    return mm(rz,ry),[.045*math.cos(2*math.pi*t),.028*math.sin(2*math.pi*t),.018*math.sin(4*math.pi*t)]

def motion(out,t=.16,frame_path=None):
    s=SVG('同一段运动，两个不同的观察坐标系',True)
    s.text(48,108,'原点和轴向不同；Controller 与 Hand 之间的固定安装关系始终不变。',24,s.muted)
    s.text(48,182,'输入：Controller pose',30)
    s.text(870,182,'转换后：Hand pose',30)
    R,p=pose(t)
    for center,which in [([390,595],'C'),([1195,595],'H')]:
        scene=Scene(s,center,2650,R,p)
        grid=Scene(s,center,2650)
        grid.grid()
        origin=CH if which=='C' else [0,0,0]
        basis=RHC if which=='C' else I
        trace=[]
        for j in range(97):
            rj,pj=pose(j/96)
            trace.append(grid.point(add(mv(rj,origin),pj)))
        for a,b in zip(trace,trace[1:]): s.line(a,b,'#8bcde4' if which=='C' else '#ecb87b',3,opacity=.8)
        for tj in [0,.33,.66]:
            rj,pj=pose(tj)
            ghost=Scene(s,center,2650,rj,pj)
            ghost.axes(origin,basis,.027,False,3,.4)
        scene.body(.35)
        scene.axes(origin,basis,.065)
        scene.s.circle(scene.point(origin),7,'#8bcde4' if which=='C' else '#ecb87b')
        q=scene.point(origin)
        s.text(q[0]-45,q[1]+47,which,32)
    s.arrow([733,433],[843,433],'#e0eafa',5)
    s.text(748,399,'标定',24)
    s.text(668,478,'固定的 T_C_H',22,s.muted)
    s.text(48,835,'追踪器告诉我们“手柄在哪里”',28)
    s.text(870,835,'标定后知道“手部原点和 XYZ 在哪里”',28)
    s.text(48,894,'原理演示：轨迹为合成运动，不是实测回放；曲线是各自原点的轨迹，浅色轴是历史姿态。',22,s.muted)
    s.save(frame_path or out/'controller-hand-motion.svg')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,default=Path('docs/images'))
    p.add_argument('--animation-frames',type=Path)
    args=p.parse_args(); calibration(args.output_dir); motion(args.output_dir)
    if args.animation_frames:
        for i in range(48): motion(args.output_dir,i/48,args.animation_frames/f'{i:03}.svg')
    print('Generated geometric SVGs; R orthogonality and determinant checked.')

if __name__=='__main__': main()
