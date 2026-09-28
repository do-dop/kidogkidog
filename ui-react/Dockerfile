FROM node:22-alpine AS build

WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
RUN npm run build

FROM nginx:1.27-alpine

ARG API_ORIGIN
COPY nginx.conf.template /tmp/default.conf.template
RUN sed "s|\${API_ORIGIN}|${API_ORIGIN}|g" /tmp/default.conf.template > /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html
