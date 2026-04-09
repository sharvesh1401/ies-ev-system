FROM node:20-alpine

WORKDIR /app

# Install dependencies first (caching)
COPY package.json ./
RUN npm install --legacy-peer-deps

# Copy source
COPY . .

# Expose port
EXPOSE 3000

# Run development server
CMD ["npm", "run", "dev"]
