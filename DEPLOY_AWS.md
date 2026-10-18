# Deploy Battery SOH + RUL on AWS EC2 with Docker

## 1. Push this folder to GitHub
Create a GitHub repository, e.g. `battery-soh-rul`. In this folder run:

```bash
git init
git add .
git commit -m "Prepare Battery SOH RUL app for deployment"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/battery-soh-rul.git
git push -u origin main
```

Replace `YOUR_USERNAME` with your GitHub username. If GitHub asks for authentication, use the browser sign-in flow or a personal access token; do not put credentials in the repository.

## 2. Create an EC2 instance
- AWS Console → EC2 → Launch instance.
- Name: `battery-soh-rul`.
- AMI: Ubuntu Server 24.04 LTS.
- Instance type: start with `t3.large` (2 vCPU, 8 GiB RAM) for a modest CPU-only deployment. Increase RAM/CPU if large uploads or LSTM training exhaust resources.
- Storage: 30 GiB gp3 to start.
- Create/download an SSH key pair and keep the private key safe.
- Security group inbound rules:
  - SSH TCP 22: **your current public IP only**
  - HTTP TCP 80: `0.0.0.0/0`
  - HTTPS TCP 443: `0.0.0.0/0`
  - Do not expose port 8501 publicly for the final setup.
- Launch and note the public IPv4 address.

## 3. Install Docker and clone the repo
SSH into the instance using the command AWS shows. Then run:

```bash
sudo apt-get update
sudo apt-get install -y ca-certificates curl git nginx
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo ${UBUNTU_CODENAME:-$VERSION_CODENAME}) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo systemctl enable --now docker
git clone https://github.com/YOUR_USERNAME/battery-soh-rul.git
cd battery-soh-rul
```

If your GitHub repository is private, configure GitHub authentication on the server securely before cloning. Do not embed tokens in the clone URL.

## 4. Build and run
```bash
sudo docker build -t battery-soh-rul:latest .
sudo docker run -d --name battery-soh-rul --restart unless-stopped \
  -p 127.0.0.1:8501:8501 battery-soh-rul:latest
sudo docker ps
sudo docker logs --tail=100 battery-soh-rul
```

The app is bound to localhost on the server so it is not directly exposed to the internet.

## 5. Configure Nginx
Create `/etc/nginx/sites-available/battery-soh-rul`:

```nginx
server {
    listen 80;
    server_name YOUR_DOMAIN_OR_EC2_PUBLIC_IP;

    location / {
        proxy_pass http://127.0.0.1:8501;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_read_timeout 86400;
        proxy_send_timeout 86400;
        client_max_body_size 100M;
    }
}
```

Enable it:

```bash
sudo ln -s /etc/nginx/sites-available/battery-soh-rul /etc/nginx/sites-enabled/battery-soh-rul
sudo nginx -t
sudo systemctl reload nginx
```

For the initial HTTP test, use your EC2 public IP in `server_name` and open `http://EC2_PUBLIC_IP`. For real production use, point a domain's DNS A record at the EC2 IP, replace `YOUR_DOMAIN_OR_EC2_PUBLIC_IP` with the domain, then enable HTTPS:

```bash
sudo apt-get install -y certbot python3-certbot-nginx
sudo certbot --nginx -d YOUR_DOMAIN
```

Certbot requires a domain that resolves to the instance. Do not expect a trusted HTTPS certificate for a bare EC2 IP.

## 6. Updates after pushing code
On the EC2 instance:

```bash
cd battery-soh-rul
git pull
sudo docker build -t battery-soh-rul:latest .
sudo docker rm -f battery-soh-rul
sudo docker run -d --name battery-soh-rul --restart unless-stopped \
  -p 127.0.0.1:8501:8501 battery-soh-rul:latest
```

## Production checks
- Test with a small CSV/XLSX and then a representative large file.
- Check logs with `sudo docker logs -f battery-soh-rul`.
- Watch memory/CPU during LSTM training (`free -h`, `top`).
- Uploaded files are processed by the app; this setup does not add persistent storage for uploads.
- Add authentication/access control before exposing sensitive battery datasets publicly.
- Set AWS billing alerts and stop/terminate the instance when not needed.
