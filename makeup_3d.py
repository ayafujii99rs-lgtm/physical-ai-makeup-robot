"""AI-generated portraits -> estimated 3D face surface -> robot makeup.
Geometry is a monocular estimate, not measured anatomy. No force control.
"""
from pathlib import Path
import argparse, json, urllib.request, os
os.environ["MPLBACKEND"] = "Agg"
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from scipy.spatial import Delaunay, cKDTree

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results'; OUT.mkdir(exist_ok=True)
MODEL_URL='https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task'
ROSE=np.array([196,104,126],dtype=float) # Approximation from supplied photograph, not a shade match.
TOOL=.055

class FaceSurface:
    def __init__(self, points, position):
        self.uv=np.asarray(points)[:468,:2]
        self.depth=np.asarray(points)[:468,2]
        self.center=np.mean(self.uv[[234,454]],axis=0)
        self.scale=.16/np.linalg.norm(self.uv[234]-self.uv[454])
        self.position=np.asarray(position,float)
        # x = image vertical; y = image horizontal; z = negative estimated depth.
        self.vertices=np.column_stack(((self.uv[:,1]-self.center[1])*self.scale,
                         (self.uv[:,0]-self.center[0])*self.scale,
                         -(self.depth-np.median(self.depth))*self.scale))+self.position
        self.tri=Delaunay(self.uv)
        self.tree=cKDTree(self.vertices)
        self.cheeks=self.uv[[50,280]].copy()
    def sample(self,uv):
        uv=np.asarray(uv,float)
        index=int(self.tri.find_simplex(uv))
        if index<0: raise ValueError('Brush path outside face mesh; stopping.')
        transform=self.tri.transform[index]
        bary=transform[:2]@(uv-transform[2]);weights=np.r_[bary,1-bary.sum()]
        ids=self.tri.simplices[index];a,b,c=self.vertices[ids]
        normal=np.cross(b-a,c-a);normal/=np.linalg.norm(normal)
        if normal[2]<0:normal=-normal
        return weights@self.vertices[ids],normal
    def export_obj(self,path):
        # OBJ triangle winding must face upwards in world space.
        lines=['# AI estimated anterior face surface; world units metres']
        lines += ['v '+' '.join(map(str,p)) for p in self.vertices]
        lines += [f'vt {u} {1-v}' for u,v in self.uv]
        for ids in self.tri.simplices:
            a,b,c=self.vertices[ids]
            if np.cross(b-a,c-a)[2]<0: ids=ids[::-1]
            lines.append('f '+' '.join(f'{i+1}/{i+1}' for i in ids))
        path.write_text('\n'.join(lines))
    def nearest_uv(self,point):
        # Approximate projection is refined by planar barycentric interpolation.
        u=(point[1]-self.position[1])/self.scale+self.center[0]
        v=(point[0]-self.position[0])/self.scale+self.center[1]
        return np.array([u,v])

def quaternion_z(axis):
    z=np.array([0.,0.,1.]);axis=np.asarray(axis);axis=axis/np.linalg.norm(axis)
    dot=float(z@axis)
    if dot<-.999999:return [1.,0.,0.,0.]
    q=np.r_[np.cross(z,axis),1+dot];return (q/np.linalg.norm(q)).tolist()

def detect(images):
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    path=ROOT/'face_landmarker.task'
    if not path.exists():urllib.request.urlretrieve(MODEL_URL,path)
    options=vision.FaceLandmarkerOptions(base_options=python.BaseOptions(model_asset_path=str(path)),num_faces=1,
            min_face_detection_confidence=.5,min_face_presence_confidence=.5)
    points=[]
    with vision.FaceLandmarker.create_from_options(options) as detector:
        for i,im in enumerate(images):
            result=detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB,data=np.asarray(im,dtype=np.uint8).copy()))
            if len(result.face_landmarks)!=1:raise RuntimeError(f'Face {i+1} not detected. Use a frontal adult portrait.')
            points.append(np.array([[p.x,p.y,p.z] for p in result.face_landmarks[0]]))
    return points

def make_compact_texture():
    im=Image.new('RGB',(512,320),(25,25,28));d=ImageDraw.Draw(im)
    d.rounded_rectangle((15,15,497,305),radius=22,fill=(182,186,191))
    d.rectangle((34,34,320,286),fill=tuple(ROSE.astype(int)))
    d.rectangle((345,34,478,286),fill=(18,18,20))
    d.text((110,118),'Dior',fill=(168,80,102),font_size=50)
    d.text((350,247),'CD',fill=(215,216,219),font_size=30)
    im.save(OUT/'compact_reference_style.png');return im

def run(atlas,smoke=False):
    import pybullet as p, pybullet_data, imageio.v2 as iio
    from importlib.metadata import version
    original=Image.open(atlas).convert('RGB');w,h=original.size
    images=[original.crop((col*w//2,row*h//2,(col+1)*w//2,(row+1)*h//2)).resize((512,512)) for row in range(2) for col in range(2)]
    points=detect(images)
    surfaces=[FaceSurface(x,[.57+dx,dy,.32]) for x,(dx,dy) in zip(points,[(0,0),(.025,.035),(-.025,-.03),(.02,-.025)])]
    for i,(im,s) in enumerate(zip(images,surfaces)):
        im.save(OUT/f'portrait_{i+1}.png');s.export_obj(OUT/f'face_{i+1}.obj')
    # Baseline: same first portrait and same world-space path for all later trials.
    reference=surfaces[0]
    p.connect(p.DIRECT);p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0,0,-9.81);p.setTimeStep(1/240)
    p.loadURDF('plane.urdf');robot=p.loadURDF('kuka_iiwa/model.urdf',useFixedBase=True)
    def box(half,pos,color):
        return p.createMultiBody(0,-1,p.createVisualShape(p.GEOM_BOX,halfExtents=half,rgbaColor=color),basePosition=pos)
    box([.17,.15,.015],[.57,0,.245],[.19,.21,.25,1])
    # Compact is an illustrative black/silver case with a rose pan, not an exact CAD replica.
    box([.065,.05,.009],[.44,-.26,.27],[.06,.06,.07,1])
    box([.061,.046,.002],[.44,-.26,.281],[.72,.74,.77,1])
    box([.038,.039,.002],[.424,-.26,.284],[*(ROSE/255),1])
    box([.014,.039,.002],[.48,-.26,.284],[.04,.04,.05,1])
    brush=p.createMultiBody(0,-1,p.createVisualShape(p.GEOM_CYLINDER,radius=.006,length=TOOL,
                    visualFramePosition=[0,0,TOOL/2],rgbaColor=[.06,.06,.08,1]))
    bristle=p.createMultiBody(0,-1,p.createVisualShape(p.GEOM_SPHERE,radius=.007,rgbaColor=[.77,.41,.49,1]))
    compact=make_compact_texture().resize((220,138))
    view=p.computeViewMatrix([1.02,-.82,.94],[.48,0,.32],[0,0,1])
    close=p.computeViewMatrix([.82,-.30,.65],[.57,0,.32],[0,0,1])
    proj=p.computeProjectionMatrixFOV(45,1.,.02,3)
    frame_count=0;rows=[];body=None
    video=OUT/('smoke_test.mp4' if smoke else 'makeup_3d_demo.mp4')
    writer=iio.get_writer(str(video),fps=20,codec='libx264',quality=8,macro_block_size=16)
    try:
        for index,(im,surface) in enumerate(zip(images,surfaces)):
            if smoke and index>0:break
            for mode in ['FIXED','AI']:
                if body is not None:p.removeBody(body)
                vis=p.createVisualShape(p.GEOM_MESH,fileName=str(OUT/f'face_{index+1}.obj'),rgbaColor=[1,1,1,1])
                body=p.createMultiBody(0,-1,vis)
                texture=OUT/f'painted_{index}_{mode}.png';im.save(texture)
                texid=p.loadTexture(str(texture));p.changeVisualShape(body,-1,textureUniqueId=texid)
                for j,q in enumerate([0,.4,0,-1.5,0,1.1,0]):p.resetJointState(robot,j,q)
                paint=np.zeros((512,512),float); yy,xx=np.mgrid[:512,:512]
                tracking=[];reference_error=[];surface_distance=[];hits=0;attempts=0;painted=im.copy()
                for cheek in range(2):
                    start=np.asarray(p.getLinkState(robot,6,computeForwardKinematics=True)[4])
                    steps=20 if smoke else 100
                    for f in range(steps):
                        phase=f/max(1,steps-1);draw=.20<=phase<=.85
                        t=np.clip((phase-.20)/.65,0,1)
                        offset_m=np.array([.009*np.cos(4*np.pi*t),.006*np.sin(4*np.pi*t)]) if draw else np.zeros(2)
                        desired,n=surface.sample(surface.cheeks[cheek]+offset_m/surface.scale)
                        chosen=surface if mode=='AI' else reference
                        contact,normal=chosen.sample(chosen.cheeks[cheek]+offset_m/chosen.scale)
                        lift=.003 if draw else .045
                        goal=contact+normal*(TOOL+lift)
                        if phase<.20:goal=start+(goal-start)*min(1,phase/.20)
                        orn=quaternion_z(-normal)
                        joints=p.calculateInverseKinematics(robot,6,goal.tolist(),orn,maxNumIterations=120,residualThreshold=1e-5)
                        p.setJointMotorControlArray(robot,list(range(7)),p.POSITION_CONTROL,targetPositions=joints[:7],forces=[300]*7)
                        for _ in range(12):p.stepSimulation()
                        state=p.getLinkState(robot,6,computeForwardKinematics=True)
                        actual=np.asarray(state[4]);rotation=np.asarray(p.getMatrixFromQuaternion(state[5])).reshape(3,3)
                        tip=actual+rotation@np.array([0,0,TOOL])
                        p.resetBasePositionAndOrientation(brush,actual.tolist(),state[5]);p.resetBasePositionAndOrientation(bristle,tip.tolist(),[0,0,0,1])
                        if draw:
                            attempts+=1
                            tracking.append(float(np.linalg.norm(tip-(contact+normal*.003))*1000))
                            reference_error.append(float(np.linalg.norm(tip-(desired+n*.003))*1000))
                            uv=surface.nearest_uv(tip)
                            try:
                                projected,n2=surface.sample(uv)
                                distance=float(np.linalg.norm(tip-projected));surface_distance.append(distance*1000)
                                if distance<.008:
                                    hits+=1; x,y=uv*512
                                    paint+=.035*np.exp(-((xx-x)**2+(yy-y)**2)/(2*9**2))
                                    alpha=np.minimum(paint,.42)[:,:,None]
                                    painted=Image.fromarray(np.uint8(np.asarray(im)*(1-alpha)+ROSE*alpha))
                            except ValueError:pass
                        if f%4==0:
                            p.changeTexture(texid,np.asarray(painted,dtype=np.uint8).ravel().tolist(),512,512)
                        def render(v):
                            data=p.getCameraImage(512,512,v,proj,renderer=p.ER_TINY_RENDERER)[2]
                            if isinstance(data,(bytes,bytearray)):arr=np.frombuffer(data,dtype=np.uint8)
                            else:arr=np.asarray(data,dtype=np.uint8)
                            return Image.fromarray(arr.reshape(512,512,4)[:,:,:3])
                        frame=Image.new('RGB',(1280,720),(18,23,34));d=ImageDraw.Draw(frame)
                        frame.paste(render(view),(16,88));frame.paste(render(close).resize((420,420)),(540,88))
                        frame.paste(painted.resize((272,272)),(992,88));frame.paste(compact,(1010,405))
                        d.text((24,18),'3D MAKEUP ROBOT',fill='white',font_size=30)
                        d.text((24,57),f'Portrait {index+1}/4 | {mode} | cheek {cheek+1}/2',fill=(237,166,185),font_size=20)
                        d.text((540,530),'Estimated curved face surface',fill='white',font_size=18)
                        d.text((992,375),'Generated portrait + blush',fill='white',font_size=16)
                        d.text((992,555),'Dior reference-style compact',fill=(210,210,215),font_size=15)
                        d.text((24,624),'AI: image landmarks -> 3D mesh -> surface-normal brush path -> joint control',fill='white',font_size=19)
                        d.text((24,656),'Monocular depth estimate | assumed face width 160 mm | no contact-force simulation',fill=(185,192,208),font_size=17)
                        d.text((24,687),'Pink paint is deposited from the actual simulated brush tip near the surface.',fill=(185,192,208),font_size=16)
                        writer.append_data(np.asarray(frame));frame_count+=1
                    print(f'Portrait {index+1}/4 {mode} cheek {cheek+1} done',flush=True)
                row={'portrait':index+1,'mode':mode,'tracking_error_mm':float(np.mean(tracking)),
                     'ai_reference_path_error_mm':float(np.mean(reference_error)),
                     'near_surface_frames':hits,'drawing_frames':attempts,
                     'near_surface_fraction':hits/max(1,attempts)}
                rows.append(row);print(json.dumps(row),flush=True)
                frame.save(OUT/f'preview_{index+1}_{mode}.png')
    finally:writer.close();p.disconnect()
    result={'status':'smoke_test' if smoke else 'completed','video_seconds':frame_count/20,'fps':20,
        'depth':'MediaPipe monocular relative depth; assumed face width 0.16m',
        'metrics_note':'AI reference is model-derived, NOT independently measured ground truth. No face-detection accuracy claim.',
        'product_note':'Appearance approximation based on user-supplied Dior photo; exact shade unknown.',
        'versions':{x:version(x) for x in ['mediapipe','pybullet','numpy','Pillow']},'trials':rows}
    (OUT/('smoke_results.json' if smoke else 'results.json')).write_text(json.dumps(result,indent=2))
    if not smoke:
        report='''# AI生成顔の3D表面推定を用いたチーク塗布ロボット\n\n## 目的・何をしたか\n架空の成人4人の顔画像から顔の3次元表面を推定し、ロボットのブラシが左右の頬に沿って動くシミュレーションを作成した。顔ごとに位置を調整する方式と、最初の顔の軌道を固定して使う方式を比較した。\n\n## 方法\n事前にAI生成した4枚の画像を使用。試行中に新規画像生成はしない。MediaPipe Face Landmarkerで顔特徴点を検出し、相対奥行きと顔幅160mmの仮定を用いて前面メッシュを作成した。頬周辺の経路を三角形上に補間し、面の法線に合わせてブラシ姿勢を指定。PyBulletの逆運動学と位置制御でKUKA iiwaを動かす。実際のブラシ先端が表面の8mm以内に来た場合にだけ色を重ねた。\n\n## 結果\n以下は今回の実行ログであり、知覚精度や実機の安全性を示す値ではない。AI参照軌道自体が同じモデルから得られた推定値で、独立した正解ではない。\n\n'''
        for r in rows:report+=f"- 顔{r['portrait']} / {r['mode']}: 追従誤差 {r['tracking_error_mm']:.2f}mm、AI参照軌道との差 {r['ai_reference_path_error_mm']:.2f}mm、表面近傍フレーム率 {r['near_surface_fraction']:.1%}\n"
        report+='''\n## 工夫\n同一顔について固定方式とAI方式を比較し、同じ制御系・塗布条件を使用した。ブラシの目標位置だけで塗布を描かず、物理ステップ後の実際の手先位置からブラシ先端を算出した。チークは提供されたDiorの商品写真を参考にローズ系色と黒・銀のケースを表現した。正確な商品CAD・色番号の再現ではない。\n\n## 限界・今後\n顔前面のみの推定メッシュであり、頭蓋全体の復元や精密スキャンではない。顔モデル・ブラシは可視化用で、顔との接触力、皮膚変形、圧力制御、障害物回避の検証はしていない。曲面追従の概念実証として評価する。今後は独立した3D正解データで評価し、力覚センサーを用いた接触制御を追加する。前の合成図形版で得られた6.45mm等の数値は本実験には流用しない。\n\n## 出典\nhttps://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker/python\nhttps://github.com/bulletphysics/bullet3\n画像生成・コード作成支援: ChatGPT。\n'''
        (OUT/'report_draft.md').write_text(report,encoding='utf-8')
    print('DONE: '+str(video),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--atlas',required=True);parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args();run(args.atlas,args.smoke)
