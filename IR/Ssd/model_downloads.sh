#!/bin/bash

wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1oLK1wMX9NPX5yF9gz_8tGFi5hSR8QGBz' -O ssd_coco.tar.xz
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1o2NOTWZ12YaCvOO6chg4cIg04D3Kl_hL' -O ssdlite_coco.tar.xz

tar -xvf ssd_coco.tar.xz
tar -xvf ssdlite_coco.tar.xz
rm ssd_coco.tar.xz
rm ssdlite_coco.tar.xz

