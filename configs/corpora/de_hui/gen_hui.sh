#!/bin/bash

rm de_hui_*.yaml

for SPEAKER in Alexandra_Bogensperger Algy_Pug Anka Availle Bernd_Ungerer Boris Capybara caromopfen Christian_Al-Kadi ClaudiaSterngucker Crln_Yldz_Ksr Dirk_Weber DomBombadil Eki_Teebi ekyale Elli Eva_K fantaeiner Friedrich Frown Hokuspokus Igor_Teaforay Imke_Grassl Ingo_Breuer Jessi Julia_Niedermaier Kalynda KarinM Karlsson keltoi Klaus_Neubauer lorda LyricalWB marham63 Martin_Harbecke Monika_M._C Ohrbuch OldZach PeWaOt Ragnar Ramona_Deininger-Schnabel Rebecca_Braunert-Plunkett RenateIngrid Rhigma Robert_Steiner Rogthey schrm Sebastian Silmaryllis Sonia Sonja Tanja_Ben_Jeroud Victoria_Asztaller ; do

    echo $SPEAKER

    cat hui_tmpl.yaml.tmpl | sed "s/SPEAKER/$SPEAKER/g" > de_hui_${SPEAKER}.yaml

done

