#!/bin/bash
# Re-download the truncated chunk 11 and the tail of the third I-80 period.
cd "$(dirname "$0")"
cols="vehicle_id,frame_id,total_frames,global_time,local_x,local_y,global_x,global_y,v_length,v_width,v_class,v_vel,v_acc,lane_id,preceding,following,space_headway,time_headway"
t0=1113433135000; t1=1113438360000; n=16
step=$(( (t1-t0)/n + 1 ))
get() {  # name a b
  curl -s -m 3000 -G "https://data.transportation.gov/resource/8ect-6jqj.csv" \
    --data-urlencode "\$select=$cols" --data-urlencode "\$where=location='i-80' AND global_time >= $2 AND global_time < $3" \
    --data-urlencode "\$limit=2000000" -o "$1"
}
a=$((t0+11*step)); b=$((a+step))
get chunk_11.csv $a $b &
get chunk_16.csv 1113438360000 1113438550000 &
get chunk_17.csv 1113438550000 1113438800000 &
wait
for f in chunk_11.csv chunk_16.csv chunk_17.csv; do echo "$f $(wc -l < $f)"; done
echo done
