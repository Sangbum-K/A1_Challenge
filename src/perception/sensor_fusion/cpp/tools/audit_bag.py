"""Read actual recorded schemas/status/time without starting drivers or publishing."""
import argparse
import json
from pathlib import Path
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

def main():
    p=argparse.ArgumentParser();p.add_argument('bag');p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();reader=rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=args.bag,storage_id='mcap'),rosbag2_py.ConverterOptions('cdr','cdr'))
    topics={t.name:t.type for t in reader.get_all_topics_and_types()}
    wanted=['/lidar_points','/gsof_client/gsof/ins_solution_49']
    reader.set_filter(rosbag2_py.StorageFilter(topics=wanted));samples={}
    while reader.has_next() and len(samples)<len(wanted):
        topic,data,received=reader.read_next()
        if topic in samples:continue
        m=deserialize_message(data,get_message(topics[topic]));stamp=m.header.stamp.sec+m.header.stamp.nanosec/1e9
        record=dict(frame=m.header.frame_id,stamp=stamp,receive_minus_stamp_s=received/1e9-stamp)
        if topic=='/lidar_points':
            record.update(width=m.width,height=m.height,point_step=m.point_step,row_step=m.row_step,
                          fields=[dict(name=f.name,offset=f.offset,type=f.datatype,count=f.count) for f in m.fields])
        else:
            record.update(velocity_ned=[m.velocity.north,m.velocity.east,m.velocity.down],heading_deg=m.heading,
                          imu_alignment=m.status.imu_alignment,gnss_status=m.status.gnss)
        samples[topic]=record
    result=dict(bag=args.bag,topics=topics,samples=samples,note='Historical recorded evidence, not current hardware status or live calibration.')
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(samples,indent=2))

if __name__=='__main__':main()
