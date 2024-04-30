#!/bin/bash

rm de_cv_*.yaml

for SPEAKER in `seq 0 702` ; do

    echo $SPEAKER

    cat cv_tmpl.yaml.tmpl | sed "s/SPEAKER/speaker_$SPEAKER/g" > de_cv_speaker_${SPEAKER}.yaml

done

