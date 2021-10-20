#!/bin/bash
#
# sudo docker build -t stylegan:0.3 .
# sudo docker tag stylegan:0.3 nvcr.io/nvidian/ct-toronto-ai/stylegan:0.3
# sudo docker push nvcr.io/nvidian/ct-toronto-ai/stylegan:0.3

sudo docker build -t ggstylegan_release:0.1 .
sudo docker tag ggstylegan_release:0.1 nvcr.io/nvidian/ct-toronto-ai/ggstylegan_release:0.1
sudo docker push nvcr.io/nvidian/ct-toronto-ai/ggstylegan_release:0.1

#./scripts/september/run_ngc_sep16.sh
