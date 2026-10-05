# Stage 1: remote video worker lab

This lab connects a separate Cloud Run API to RabbitMQ and a Celery worker on
one Compute Engine VM. The edge simulator still runs locally and uploads test
chunks to a **lab** GCS bucket. Do not point this lab at the existing staging
database, ChromaDB collection, or bucket.

## What is included

- `ops/ansible/`: roles that install Docker and deploy RabbitMQ, the Cloud SQL
  Auth Proxy, and the Celery worker (Compose file in
  `roles/worker/files/compose.yaml`) on an Ubuntu 24.04 VM.
- `/upload`: optional `X-Upload-Token` check, GCS object validation, a returned
  Celery task ID, and a 503 response if RabbitMQ is unavailable.

The VM, subnet, firewall, service accounts, Cloud SQL instance, ChromaDB
server, bucket, and lab Cloud Run API must be created separately. The playbook
changes only the selected worker VM. Jenkins and Cloud Run rollout automation
are later stages.

## Prepare the lab resources

1. Create a dedicated worker VM with a fixed private IP in the same VPC as the
   lab Cloud Run API and ChromaDB. Give it the `cloud-run-api` network tag so
   the existing firewall rule lets it reach ChromaDB on port 8000. Attach a VM
   service account that can read and write objects in the lab bucket, connect to
   the lab Cloud SQL instance, and pull the worker image from Artifact Registry.
   Give the VM the `cloud-platform` access scope; the IAM roles still limit its
   actual permissions. No service-account JSON key is needed on the VM.
2. Allow inbound TCP 5672 to the worker VM only from the lab Cloud Run VPC
   egress subnet. The Compose file binds this port to the VM's private IP.
   RabbitMQ management is bound to VM loopback on 15672; reach it through SSH
   forwarding when needed. Do not open these ports to the public internet.
3. Prepare a separate lab bucket, MySQL database, and ChromaDB collection.
   Use a separate ChromaDB server before testing Chroma failures.
4. Build `api/Dockerfile` from the desired commit and push the image to Artifact
   Registry. The same image contains the API and worker code; Compose overrides
   its default command to start Celery. Use an immutable commit tag or digest.

## Deploy the worker VM with Ansible

Layout (`ops/ansible/`):

```text
ansible.cfg                       roles path, vault identity, pipelining
playbooks/site.yml                serial: 1, roles: docker, worker
inventory/lab.ini                 real host (ignored by Git; copy lab.example.ini)
inventory/group_vars/all/vars.yml non-secret settings, worker_image_tag
inventory/group_vars/all/vault.yml  Ansible Vault (vault_* secrets)
roles/docker                      Docker Engine + Compose plugin
roles/worker                      env files, Compose, deploy, health check, rollback
```

One-time setup on the control machine:

```bash
cd ops/ansible
ansible-galaxy collection install -r requirements.yml
cp inventory/lab.example.ini inventory/lab.ini        # fill in host and SSH user
echo '<vault password>' > .vault_pass.lab && chmod 600 .vault_pass.lab
ansible-vault edit inventory/group_vars/all/vault.yml  # vault_mysql_password, ...
```

Deploy (set `worker_image_tag` in `vars.yml` or pass it with `-e`):

```bash
ansible-playbook playbooks/site.yml --check --diff          # preview
ansible-playbook playbooks/site.yml -e worker_image_tag=<tag>
```

What the `worker` role does: writes root-only env files, logs Docker in to
Artifact Registry with the VM's own short-lived token, runs
`docker compose up --pull always --wait`, then **verifies** the worker
(metrics endpoint on loopback `8002` answers and RabbitMQ lists a consumer).
Only after that does it record the image in `.last_good_image`. If any step in
the deploy block fails, the `rescue` section re-renders `.env` with the last
good image, starts it, and fails the run, so a pipeline sees a failed build
and the VM keeps running the previous version. A re-run with nothing changed
reports `changed=0`.

The worker and Cloud SQL proxy use the VM service account at runtime. The
proxy is reachable only inside the Compose network, so the worker uses
`MYSQL_HOST=cloudsql-proxy`. It connects to the Cloud SQL instance over its
public IP (the instance has no private IP; the proxy authenticates with the VM
service account and encrypts the connection) and does not use the local
`gcp-cloudsql-ca.pem` file.

RabbitMQ's `RABBITMQ_DEFAULT_*` values create the initial account on an empty
data volume. Changing those values after RabbitMQ has initialized does not
rotate the existing account; rotate it explicitly in RabbitMQ.

## Connect the lab Cloud Run API

Deploy a **separate lab service** using the same API image. Set its Direct VPC
egress to the lab subnet and `private-ranges-only`. Supply these settings:

| Setting | Value |
| --- | --- |
| `CELERY_BROKER_URL` | Secret Manager secret with `amqp://USER:PASSWORD@VM_PRIVATE_IP:5672/VHOST` |
| `UPLOAD_API_TOKEN` | A separate Secret Manager secret shared with the local simulator |
| `MYSQL_UNIX_SOCKET`, `MYSQL_USER`, `MYSQL_DATABASE`, `MYSQL_PASSWORD` | Lab Cloud SQL values |
| `CHROMA_HOST`, `CHROMA_PORT`, `CHROMA_COLLECTION` | Lab ChromaDB private address, `8000`, and lab collection |
| `GCS_BUCKET_NAME` | Lab bucket |

The Cloud Run service account needs access to the two secrets, the lab GCS
objects, and Cloud SQL. `UPLOAD_API_TOKEN` is intentionally optional in the
code so existing local environments keep working; **set it on the lab API**
before accepting requests. The public API currently includes `/upload`.

The existing GitHub Actions workflow still deploys normal staging after a
successful `main` build. Keep the lab API on a different service name so this
workflow cannot replace its settings.

## One-chunk acceptance check

1. On the VM, confirm RabbitMQ is healthy and the Celery worker is running.
2. Set `GCS_BUCKET_NAME` and `UPLOAD_API_TOKEN` for the **lab** in an ignored
   local `.env.lab`. Authenticate local GCS access with Application Default
   Credentials.
3. Run a short test video through the simulator with the lab API URL:

   ```bash
   python -m dotenv -f .env.lab run -- \
     python -m simulator.edge_simulator path/to/short-video.mp4 \
     --server-url https://LAB_API_URL --interval 0
   ```

4. Check that `/upload` returns a task ID, the RabbitMQ queue drains, the
   worker reports completion, and the lab GCS frames, ChromaDB records, and
   MySQL rows correspond to the same video ID. Check a search result in the
   lab API last.

If the VM is stopped, RabbitMQ is also stopped and the API cannot accept new
analysis work. For this one-VM lab, accept uploads only while the VM is on.

## Monitoring

The worker serves Prometheus metrics on VM loopback `8002`
(`kidog_pipeline_*`), and RabbitMQ serves its own on loopback `15692`
(queue depth, unacked messages, consumers). Neither is exposed publicly and
nothing scrapes them automatically. To watch a run from your machine:

```bash
monitoring/lab-tunnel.sh &                       # SSH tunnel to the VM
docker compose -f monitoring/compose.lab.yml up -d
```

Prometheus is then on http://127.0.0.1:9091 and Grafana on
http://127.0.0.1:3001 (dashboards "실습 큐와 worker" and "영상 처리").

## Failure handling

`process_chunk` acks after it finishes (`acks_late`) and requeues if the worker
process dies, so a killed worker does not lose the message. Errors are retried
3 times with backoff (`CHUNK_MAX_RETRIES`); a missing GCS object is not
retried. Re-runs are safe because MySQL results are committed together with a
`processed_chunks` marker. Limits: a chunk that always kills the worker is
redelivered indefinitely (no delivery limit is set), and a single-VM RabbitMQ
is lost if the VM or its disk is lost.
