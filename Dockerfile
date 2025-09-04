FROM node:22-slim

WORKDIR /opt/sqdgn-pipes

# Install dependencies for better error handling and debugging
RUN apt-get update && apt-get install -y \
    sqlite3 \
    git \
    python3 \
    build-essential \
    netcat-traditional \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY package.json yarn.lock .yarnrc.yml ./
RUN corepack enable
RUN yarn install

COPY . .

# Create directories for metadata and cache
RUN mkdir -p /opt/sqdgn-pipes/metadata /opt/sqdgn-pipes/cache

# Add startup script
COPY docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]

# docker build -t mo4islona/sqdgn-pipes:latest .