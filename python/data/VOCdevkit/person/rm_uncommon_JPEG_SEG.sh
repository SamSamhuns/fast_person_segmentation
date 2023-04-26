#!/bin/bash

for file in JPEGImages/*
do
filestr=`basename $file`
# extension="${filestr##*.}"
filename="${filestr%.*}"
if [ `ls SegmentationClass | grep -c $filename.png` -eq 0 ]; then
echo "$file not exists in SegmentationClass, removing $file now"
rm $file
fi
done

for file in SegmentationClass/*
do
filestr=`basename $file`
# extension="${filestr##*.}"
filename="${filestr%.*}"
if [ `ls JPEGImages | grep -c $filename.jpg` -eq 0 ]; then
echo "$file not exists in JPEGImages, removing $file now"
rm $file
fi
done
