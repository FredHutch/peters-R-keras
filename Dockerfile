# build me as fredhutch/peters-r-keras:latest

FROM fredhutch/gpu-keras-tidyverse

RUN apt-get update -y

# git, curl, and unzip are already installed

RUN pip install awscli

RUN curl -L https://raw.githubusercontent.com/FredHutch/url-fetch-and-run/master/fetch-and-run/fetch_and_run.sh > /usr/local/bin/fetch_and_run.sh && \
    chmod a+x /usr/local/bin/fetch_and_run.sh

ENTRYPOINT ["/usr/local/bin/fetch_and_run.sh"]



