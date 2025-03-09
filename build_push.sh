docker builder prune --filter "until=12h" --force
docker build -t docker.roshan-ai.ir/kashf-train/kashf_train_job:1.0.15 --build-arg CACHE_BUST=$(date +"%Y%m%d%H") .
docker push docker.roshan-ai.ir/kashf-train/kashf_train_job:1.0.15
kubectl apply -f job-3090.yml -n kashf-train
kubectl get pods -n kashf-train -o wide
# ####kubectl delete job gpu-kashf-train-job -n kashf-train
### kubectl logs -n kashf-train -f gpu-kashf-train-job6-29mfs
### kubectl get pods -n kashf-train -o wide

