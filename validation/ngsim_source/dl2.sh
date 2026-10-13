#!/bin/bash
# Download NGSIM I-80 trajectories from data.transportation.gov in time-range chunks.
cd "$(dirname "$0")"
cols="vehicle_id,frame_id,total_frames,global_time,local_x,local_y,global_x,global_y,v_length,v_width,v_class,v_vel,v_acc,lane_id,preceding,following,space_headway,time_headway"
t0=1113433135000; t1=1113438360000; n=16
step=$(( (t1-t0)/n + 1 ))
for i in $(seq 0 $((n-1))); do
  a=$((t0+i*step)); b=$((a+step))
  curl -s -m 1800 -G "https://data.transportation.gov/resource/8ect-6jqj.csv" \
    --data-urlencode "\$select=$cols" --data-urlencode "\$where=location='i-80' AND global_time >= $a AND global_time < $b" \
    --data-urlencode "\$limit=2000000" -o chunk_$i.csv &
  if (( (i+1) % 4 == 0 )); then wait; fi
done
wait
for f in chunk_*.csv; do echo "$f $(wc -l < $f)"; done
echo done
