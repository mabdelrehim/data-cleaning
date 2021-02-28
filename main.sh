DATANAME=$1
src=$2
tgt=$3
DATADIR=/mnt/disk512/data/indo-european-ar/$src

if test  -f $DATADIR/$DATANAME.$tgt-$src.$src.clean ; then
  echo "skipping ${DATANAME}.${tgt}-${src} pairs."
else
python3 main.py --data_dir $DATADIR  \
-p $DATANAME.${tgt}-${src} \
-s $src -t $tgt 
fi

if test -f $DATANAME.$tgt-$src.$tgt.clean_filtered ; then
  echo "skipping using laser on ${DATANME.}.${tgt}-${src}"
else
python3 laser_filter.py  --data_dir $DATADIR \
-p $DATANAME.$tgt-$src  \
-s $src -t $tgt  \
--laser_thresh 70 
fi

