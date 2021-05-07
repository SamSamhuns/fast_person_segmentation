for file in clip_img_OUT/*
do
filestr=`basename $file`
# extension="${filestr##*.}"
filename="${filestr%.*}"
if [ `ls matting_OUT | grep -c $filename.png` -eq 0 ]; then
echo "$file not exists in matting_OUT, removing $file now"
rm $file
fi
done

for file in matting_OUT/*
do
filestr=`basename $file`
# extension="${filestr##*.}"
filename="${filestr%.*}"
if [ `ls clip_img_OUT | grep -c $filename.jpg` -eq 0 ]; then
echo "$file not exists in clip_img_OUT, removing $file now"
rm $file
fi
done
