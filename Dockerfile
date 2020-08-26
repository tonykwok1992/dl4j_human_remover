FROM alpine:latest AS model
RUN apk --no-cache add --update ca-certificates openssl wget tar && update-ca-certificates
RUN wget http://download.tensorflow.org/models/deeplabv3_mnv2_pascal_train_aug_2018_01_29.tar.gz
RUN mkdir -p /etc/model/
RUN tar xvzf deeplabv3_mnv2_pascal_train_aug_2018_01_29.tar.gz -C /etc/model/ --strip=1

FROM gradle:6.6.0-jdk11 AS cache
RUN mkdir -p /home/gradle/cache_home
ENV GRADLE_USER_HOME /home/gradle/cache_home
COPY build.gradle settings.gradle gradlew /home/gradle/src/
WORKDIR /home/gradle/src
RUN gradle build

FROM gradle:6.6.0-jdk11 AS build
COPY --from=cache /home/gradle/cache_home /home/gradle/.gradle
COPY --chown=gradle:gradle . /home/gradle/src
WORKDIR /home/gradle/src
RUN gradle build

FROM openjdk:11-jre-slim
RUN mkdir /app
COPY --from=build /home/gradle/src/build/distributions/remove_background_dl4j.tar /app/remove_background_dl4j.tar
WORKDIR /app
RUN tar -xvf remove_background_dl4j.tar
RUN rm remove_background_dl4j.tar
RUN mkdir -p /etc/model/
COPY --from=model /etc/model/frozen_inference_graph.pb /etc/model/model.pb
ENTRYPOINT ["/app/remove_background_dl4j/bin/remove_background_dl4j"]