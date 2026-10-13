# Kubernetes (cs-astep02, namespace `project-5-str`)

| File | What it runs | Address (AAU VPN) |
|---|---|---|
| `api.yaml` | Our C# API (`ghcr.io/rasmusbastian/trajectory-api`): Deployment, Service and Ingress | <http://api.project-5-str.cs-astep02.srv.aau.dk/health> |
| `elessar/` | The previous group's Elessar (`sebylizz/p5-elessar`) | <http://project-5-str.cs-astep02.srv.aau.dk/docs> |

The host names are not in DNS yet. Until they are, add them to your hosts file
(`C:\Windows\System32\drivers\etc\hosts` on Windows, `/etc/hosts` on macOS/Linux; needs admin rights):

```
192.38.60.66  project-5-str.cs-astep02.srv.aau.dk
192.38.60.66  api.project-5-str.cs-astep02.srv.aau.dk
```

## Applying changes (on the server)

`sudo k3s kubectl` is admin over the whole cluster, which other groups also use, so always pass `-n project-5-str`
or use these files, which have the namespace set. `diff` first shows what would change, without changing anything:

```bash
git clone https://github.com/RasmusBastian/Similar-Trajectory-Retrieval-gruppe-12.git && cd Similar-Trajectory-Retrieval-gruppe-12
sudo k3s kubectl diff -f k8s/ -R
sudo k3s kubectl apply -f k8s/ -R
```

Useful commands:

```bash
sudo k3s kubectl get pods,svc,ingress -n project-5-str
sudo k3s kubectl logs -n project-5-str deploy/trajectory-api
sudo k3s kubectl rollout restart -n project-5-str deploy/trajectory-api   # pull the newest :latest image
```
