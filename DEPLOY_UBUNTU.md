# Deploying Kanini Hackathon ML API to Ubuntu Server

This guide provides step-by-step instructions to deploy the FastAPI application to an Ubuntu server using Docker and Docker Compose.

## Prerequisites

- An Ubuntu Server (20.04 or 22.04 recommended).
- SSH access to the server with `sudo` privileges.
- Port `8000` allowed on the server's firewall.

## Step 1: Connect to Your Server

Open your terminal and SSH into your server:

```bash
ssh user@your_server_ip
```

## Step 2: Update System and Install Docker

Run the following commands to update your system and install Docker:

```bash
# Update package list and upgrade packages
sudo apt update && sudo apt upgrade -y

# Install Docker and Docker Compose
sudo apt install docker.io docker-compose -y

# Start and enable Docker
sudo systemctl start docker
sudo systemctl enable docker

# Add your user to the docker group (to run docker without sudo)
sudo usermod -aG docker $USER

# Apply group changes (you may need to log out and log back in, or run this)
newgrp docker
```

## Step 3: Deployment

### Option A: Clone from Git (Recommended)

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd kanini-hackathon-ML
    ```

2.  **Build and Run:**
    ```bash
    docker-compose up -d --build
    ```
    - `-d` runs the container in detached mode (background).
    - `--build` forces a rebuild of the image.

### Option B: Copy Files Manually

If you don't use Git on the server, you can copy the files from your local machine using `scp`:

```bash
# Run this from your LOCAL machine inside the project directory
scp -r . user@your_server_ip:~/kanini-hackathon-ML
```

Then on the server:
```bash
cd ~/kanini-hackathon-ML
docker-compose up -d --build
```

## Step 4: Verify Deployment

Check if the container is running:

```bash
docker-compose ps
```

View the logs to ensure the application started correctly:

```bash
docker-compose logs -f
```

(Press `Ctrl+C` to exit logs)

## Step 5: Access the Application

Open your web browser and navigate to:

-   **Swagger UI:** `http://<your_server_ip>:8000/docs`
-   **Health Check:** `http://<your_server_ip>:8000/health`

## Common Commands

-   **Stop the application:**
    ```bash
    docker-compose down
    ```
-   **Restart the application:**
    ```bash
    docker-compose restart
    ```
-   **View logs:**
    ```bash
    docker-compose logs -f
    ```

## Troubleshooting

-   **Permission Denied:** Ensure you added your user to the `docker` group (Step 2).
-   **Port Already in Use:** Check if another service is using port 8000. You can change the port in `docker-compose.yml`.
