FROM gradle:6.6.0-jdk11 AS build
COPY --chown=gradle:gradle . /home/gradle/src
WORKDIR /home/gradle/src
RUN gradle build --no-daemon

FROM openjdk:11-jre-slim
RUN mkdir /app
COPY --from=build /home/gradle/src/build/distributions/remove_background_dl4j.tar /app/remove_background_dl4j.tar
WORKDIR /app
RUN tar -xvf remove_background_dl4j.tar
RUN rm remove_background_dl4j.tar
ENTRYPOINT ["/app/remove_background_dl4j/bin/remove_background_dl4j"]