## Documentation

 
Guide to using AWS batch at Fred Hutch:
 
https://fredhutch.github.io/aws-batch-at-hutch-docs/
 
I have already given you access to Batch.
 
Link to github repository:
 
https://github.com/FredHutch/peters-R-keras
 
 
To remove files:

``` 
dtenenba@C02S72TRFVH8 ~ $ aws s3 ls s3://fh-pi-peters-u/
PRE External/
PRE Imputation_Data/
PRE test/
2018-08-15 10:36:35 45 score.csv
2018-08-14 10:05:35 47721977 x_test.csv
2018-08-14 10:04:01 748362884 x_train.csv
dtenenba@C02S72TRFVH8 ~ $ aws s3 rm s3://fh-pi-peters-u/x_test.csv
delete: s3://fh-pi-peters-u/x_test.csv
dtenenba@C02S72TRFVH8 ~ $ aws s3 rm s3://fh-pi-peters-u/x_train.csv
delete: s3://fh-pi-peters-u/x_train.csv
dtenenba@C02S72TRFVH8 ~ $

```

I suggest uploading to a distinct "path" or "folder" for each run.
 
Copy the URLs (which start out with "s3://" printed out by the "aws s3 cp' command.You will need them later.
 
Documentation for submitting a job:
 
https://docs.aws.amazon.com/cli/latest/reference/batch/submit-job.html
 
To submit a job:
 
Copy the file `job-example.json` in this repository to
`job.json`:

```
cp job-example.json job.json
```

Then edit `job.json` to be the way you want it.

**NOTE:** If you want to use `R`, set the value of 
`jobDefinition` to `peters-R-keras:2`.
If you want to use `python`, set it to
`peters-python-keras:1`.


 
aws batch submit-job --cli-input-json file://job.json
{
"jobName": "test-job-20180816",
"jobId": "db3b7e51-7ad6-4aee-8561-805ffb1c851b"
}


Monitoring a running job



From the command line:


aws batch describe-jobs --jobs db3b7e51-7ad6-4aee-8561-805ffb1c851b
 
Or the web UI:
 
https://batch-dashboard.fhcrc.org/#jobs_header
 
 
 
