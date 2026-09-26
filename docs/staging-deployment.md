# Staging deployment

For normal releases, merge a pull request into `main`. CI runs first; when it succeeds, `Deploy staging` automatically deploys that exact commit. The **GitHub > Actions > Deploy staging > Run workflow** button remains available for a manual redeploy.

## One-time GCP setup

1. Enable Cloud Run, Artifact Registry, Cloud Build, Secret Manager, Cloud SQL Admin, and Compute Engine APIs.
2. Create a Docker Artifact Registry repository in `asia-northeast3`.
3. Configure the Chroma VM to accept TCP 8000 on its **internal** IP, and create a firewall rule allowing TCP 8000 from the `cloud-run-api` network tag to the VM's `chroma-server` network tag. Do not allow public access to port 8000.
4. Store the MySQL password in Secret Manager and grant the deploy service account access to it. The Cloud Run runtime service account needs Cloud SQL Client and Secret Manager Secret Accessor.
5. Configure GitHub OIDC / Workload Identity Federation for `do-dop/kidogkidog`; do not upload a GCP service-account JSON key to GitHub.

## GitHub configuration

Create the `staging` GitHub Environment, then set the following repository variables.

| Variable | Example |
| --- | --- |
| `GCP_PROJECT_ID` | `kidogkidog-509506` |
| `GCP_REGION` | `asia-northeast3` |
| `GAR_REPOSITORY` | `kidog` |
| `CLOUD_RUN_API_SERVICE` | `kidog-api-staging` |
| `CLOUD_RUN_WEB_SERVICE` | `kidog-web-staging` |
| `CLOUD_RUN_RUNTIME_SERVICE_ACCOUNT` | `kidog-run-staging@kidogkidog-509506.iam.gserviceaccount.com` |
| `CLOUD_SQL_INSTANCE` | `kidogkidog-509506:asia-northeast3:kidog-mysql` |
| `VPC_NETWORK` | `default` |
| `VPC_SUBNET` | `default` |
| `CHROMA_INTERNAL_IP` | `10.178.0.2` |
| `MYSQL_USER` | application DB user |
| `MYSQL_DATABASE` | application database name |
| `MYSQL_PASSWORD_SECRET` | Secret Manager secret name |

Set these GitHub Environment secrets:

| Secret | Value |
| --- | --- |
| `GCP_WIF_PROVIDER` | Workload Identity Provider resource name |
| `GCP_DEPLOY_SERVICE_ACCOUNT` | deploy service-account email |
`MYSQL_UNIX_SOCKET` is injected by the workflow, so the API reaches Cloud SQL through Cloud Run's Cloud SQL connector rather than a public IP allowlist.

## Initial search-quality testing scope

The workflow deploys the API and web UI only. The API scales down to zero when it is idle; it has an 8 GiB/2 CPU limit only while handling a request. The extra memory is required while the CLIP model is loaded.

ChromaDB remains on the existing VM so every tester queries the same 377-vector collection. The API uses Cloud Run Direct VPC egress to reach the VM's internal IP; it does not use a Serverless VPC Access connector, which would keep connector VMs billed even while idle.

Celery and RabbitMQ are not deployed yet. Run the worker locally only while ingesting or re-indexing video; deployed users can still test search quality against already-indexed vectors.
