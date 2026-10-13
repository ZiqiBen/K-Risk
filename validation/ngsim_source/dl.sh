#!/bin/bash
set -e
cols="vehicle_id,frame_id,total_frames,global_time,local_x,local_y,global_x,global_y,v_length,v_width,v_class,v_vel,v_acc,lane_id,preceding,following,space_headway,time_headway"
for i in 0 1 2 3 4; do
  off=$((i*1000000))
  curl -s -m 1800 -G "https://data.transportation.gov/resource/8ect-6jqj.csv" \
    --data-urlencode "\$select=$cols" --data-urlencode "\$where=location='i-80'" \
    --data-urlencode "\$order=:id" --data-urlencode "\$limit=1000000" --data-urlencode "\$offset=$off" -o part_$i.csv
  echo "part $i $(wc -l < part_$i.csv)"
done
echo done
