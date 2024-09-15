#!/bin/bash

rm de_cvs_*.yaml

for SPEAKER in `seq 0 15204` ; do

    echo $SPEAKER

    cat cv_tmpl.yaml.tmpl | sed "s/SPEAKER/speaker_$SPEAKER/g" > de_cvs_speaker_${SPEAKER}.yaml

done

