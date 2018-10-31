#!/bin/bash

#file=$1
dir=$1

#awk 'BEGIN { max=0 } $3 > max { max=$3; seq=$1 } END { print seq; print max }' FS="," $file
#awk 'BEGIN { num=0 } $6 == 1 { num++; rts=$2; print rts } END { print num; }' FS="," $file

for f in ${dir}/*.data; do
    mv -- "$f" "${f%.data}.csv"
done
