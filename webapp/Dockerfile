# AAU CE2 AI Study Hub — static study site on port 6767
# Build from the REPO ROOT (needs exam_prep/ as content source):
#   docker build -f webapp/Dockerfile -t studyhub .
#   docker run -d --name studyhub -p 6767:6767 --restart unless-stopped studyhub
FROM node:20-alpine

WORKDIR /app

# Content source + app (marked is vendored in webapp/vendor — no npm install)
COPY exam_prep/ exam_prep/
COPY webapp/build.js webapp/server.js webapp/package.json webapp/
COPY webapp/vendor/ webapp/vendor/

# Pre-render everything into webapp/dist
RUN node webapp/build.js

EXPOSE 6767
ENV PORT=6767
CMD ["node", "webapp/server.js"]
