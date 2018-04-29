#!/bin/bash
#SBATCH --job-name=tcf
#SBATCH --time=3-23:59:59
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=204800

date

echo 'Begin Download!~~~~~~~~~~'

downloaddir='/scratch/rm885/gdrive/sync/decid/tc_files/gz'
extractdir='/scratch/rm885/gdrive/sync/decid/tc_files/tif'
regionfile='/scratch/rm885/gdrive/sync/decid/bounds/above_domain/domain_simple.shp'
wrsfile='/scratch/rm885/gdrive/sync/decid/bounds/wrs2/wrs2_descending.shp'
year=2010

python '/home/rm885/projects/decid/src/get_tc_tiles.py' $year downloaddir $extractdir $regionfile $wrsfile

echo 'End Download!~~~~~~~~~~'

echo ''

echo 'Begin Reprojection!~~~~~~~~~~'
echo '********************************************************************************************'

datadir='/scratch/rm885/gdrive/sync/decid/tc_files/tif/'
outdir='/scratch/rm885/gdrive/sync/decid/tc_files/tif_geo/'
export GDAL_DATA=/home/rm885/path/gdal

# list of input files
files=(${datadir}*.tif)
echo ${files[*]}

for f in ${files[*]}; do
   bname=$(basename $f);
   extension="${bname##*.}";
   filename="${bname%.*}";
   outf=$outdir""$filename"_geo."$extension;
   if [ -f $outf ] ; then
      rm -f $outf;
   fi;
   echo 'Reprojecting '$f' to '$outf;
   gdalwarp -overwrite -multi $f $outf -ot Byte -tr 0.00027 0.00027 -et 0.05 -t_srs 'EPSG:4326' -wo WRITE_FLUSH=YES -wo NUM_THREADS=4 ;
done

echo 'Begin Mosaic!~~~~~~~~~~'
echo '********************************************************************************************'
datadir='/scratch/rm885/gdrive/sync/decid/tc_files/tif_geo/'
outdir='/scratch/rm885/gdrive/sync/decid/tc_files/mosaic/'
mosaic=$outdir'uncert_mosaic.tif'
compmosaic=$outdir'uncert_mosaic_vis.tif'

echo 'Data folder: '$datadir
echo 'Mosaic: '$mosaic
echo 'Compressed mosaic: '$compmosaic

echo '********************************************************************************************'

# make mosaic
# spatial resolution: 30m
# data type: Byte
# background value: 0
# file format: Geotiff
gdal_merge.py -init 0 -o $mosaic -of GTiff -ps 0.00027 0.00027 -ot Byte ${files[*]}

# compress large tif file using LZW compression
# use BIGTIFF=YES for large files
gdal_translate -of GTiff -co COMPRESS=LZW -co BIGTIFF=YES $mosaic $compmosaic

# make overview (pyramid) file: gdaladdo -> gdal add overview
# this is useful if at any point ArcGIS is going to be used with this data
# this makes pyramids and will save that step with ArcGIS
gdaladdo -ro $compmosaic 2 4 8 16 32 64 128 256 --config COMPRESS_OVERVIEW LZW
echo '********************************************************************************************'
date
