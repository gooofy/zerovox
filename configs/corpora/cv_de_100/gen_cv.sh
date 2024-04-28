#!/bin/bash

rm de_cv_*.yaml

for SPEAKER in `seq 0 99` ; do

    echo $SPEAKER

    cat cv_tmpl.yaml | sed "s/SPEAKER/speaker_$SPEAKER/g" > de_cv_speaker_${SPEAKER}.yaml

done

