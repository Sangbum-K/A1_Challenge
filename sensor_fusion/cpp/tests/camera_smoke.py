"""Compare actual C++ camera preprocessing/inference with exported Python inference."""
import argparse
import subprocess
import time
import numpy as np
import torch
import rclpy
from sensor_msgs.msg import Image
from a1_fusion_cpp.msg import A2Features
from rclpy.qos import qos_profile_sensor_data

def main():
    p=argparse.ArgumentParser();p.add_argument('--binary',required=True);p.add_argument('--models',required=True)
    p.add_argument('--device',default='cpu')
    args=p.parse_args();torch.set_num_threads(2)
    torch._C._jit_set_profiling_executor(False);torch._C._jit_set_profiling_mode(False)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    rclpy.init();n=rclpy.create_node('a1_camera_cpp_test');outputs=[]
    pub=n.create_publisher(Image,'/test/image',qos_profile_sensor_data)
    sub=n.create_subscription(A2Features,'/lead_fusion/a2',outputs.append,qos_profile_sensor_data)
    proc=subprocess.Popen([args.binary,'--ros-args','-p','image_topic:=/test/image','-p','model_dir:='+args.models,'-p','device:='+args.device],stdout=subprocess.DEVNULL)
    try:
        deadline=time.monotonic()+120
        while pub.get_subscription_count()==0 and time.monotonic()<deadline:rclpy.spin_once(n,timeout_sec=.05)
        assert pub.get_subscription_count()>0,'C++ camera did not subscribe'
        image=np.zeros((720,1280,3),dtype=np.uint8)
        image[:,:,0]=np.arange(1280,dtype=np.uint16)[None,:]%256
        image[:,:,1]=97;image[:,:,2]=151
        msg=Image();msg.header.stamp=n.get_clock().now().to_msg();msg.header.frame_id='hikrobot_camera_vis'
        msg.width=1280;msg.height=720;msg.encoding='bgr8';msg.step=3840;msg.data=image.tobytes();pub.publish(msg)
        deadline=time.monotonic()+60
        next_publish=time.monotonic()+1
        while not outputs and time.monotonic()<deadline:
            rclpy.spin_once(n,timeout_sec=.05)
            if time.monotonic()>next_publish:
                pub.publish(msg)  # Best-effort image transport may drop initial discovery packets.
                next_publish=time.monotonic()+1
        assert outputs,'no C++ A2 output'
        actual=np.array(outputs[0].raw).reshape(1,40,7)
        # Initial history repeats current token; it must remain marked unready.
        assert not outputs[0].history_ready
        rgb=image[235:485,:,::-1].copy().astype(np.float32)/255
        rgb=(rgb-np.array([.485,.456,.406],np.float32))/np.array([.229,.224,.225],np.float32)
        x=torch.from_numpy(rgb.transpose(2,0,1).copy()).unsqueeze(0).to(args.device)
        tok=torch.jit.load(args.models+'/tokenizer.pt',map_location=args.device);head=torch.jit.load(args.models+'/a2_head.pt',map_location=args.device)
        with torch.inference_mode():
            t=tok(x);expected=head(t.unsqueeze(0).repeat(21,1,1,1),torch.zeros(1,2,device=args.device)).cpu().numpy()
        np.testing.assert_allclose(actual,expected,rtol=1e-4,atol=1e-5)
        print('C++ image -> A2 vs Python max abs error:',np.max(np.abs(actual-expected)))
    finally:
        proc.terminate()
        try:proc.wait(timeout=5)
        except subprocess.TimeoutExpired:proc.kill();proc.wait()
        n.destroy_node();rclpy.shutdown()

if __name__=='__main__':main()
