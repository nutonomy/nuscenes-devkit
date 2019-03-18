pipeline {

  agent none

  environment {
    PROD_IMAGE = "nuscenes:production"
  }

  stages {
    stage('Build and test'){
      agent {
        kubernetes {
          label 'nuscenes-builder'
          defaultContainer 'jnlp'
          yaml """
            apiVersion: v1
            kind: Pod
            metadata:
              labels:
                app: nuscenes
            spec:
              containers:
              - name: jnlp
                image: registry.nutonomy.com:5000/nu/jnlp-slave:3.19-1-lfs
                imagePullPolicy: Always
              - name: docker
                image: registry.nutonomy.com:5000/nu/docker-bash:latest
                command:
                - cat
                tty: true
                volumeMounts:
                - mountPath: /var/run/docker.sock
                  name: docker
              imagePullSecrets:
              - name: regcredjenkins
              volumes:
              - name: docker
                hostPath:
                  path: /var/run/docker.sock
          """
        }// kubernetes
      } // agent

      steps {
        container('docker') {
          // Build the Docker image, and then run python -m unittest inside 
          // an activated Conda environment inside of the container.
          sh """#!/bin/bash
          bash test_mini_split.sh Dockerfile_3.5
          """
        } // container

        container('docker') {
          // Build the Docker image, and then run python -m unittest inside
          // an activated Conda environment inside of the container.
          sh """#!/bin/bash
          bash test_mini_split.sh Dockerfile_3.6
          """
        } // container

        container('docker') {
          // Build the Docker image, and then run python -m unittest inside
          // an activated Conda environment inside of the container.
          sh """#!/bin/bash
          bash test_mini_split.sh Dockerfile_3.7
          """
        } // container
      }
    } // stage('Build and test')
    stage('Deploy') {
      agent {
        kubernetes {
          label 'nuscenes-' + UUID.randomUUID().toString()
          defaultContainer 'jnlp'
          yaml """
            apiVersion: v1
            kind: Pod
            metadata:
              labels:
                app: nuscenes
            spec:
              containers:
              - name: jnlp
                image: registry.nutonomy.com:5000/nu/jnlp-slave:3.19-1-lfs
                imagePullPolicy: Always
              - name: docker
                image: registry.nutonomy.com:5000/nu/docker-bash:latest
                command:
                - cat
                tty: true
                volumeMounts:
                - mountPath: /var/run/docker.sock
                  name: docker
              imagePullSecrets:
              - name: regcredjenkins
              volumes:
              - name: docker
                hostPath:
                 path: /var/run/docker.sock
          """
        }// kubernetes
      }

      when {
        branch 'master'
      }
      steps {
        // TODO: determine where to deploy Docker images.
        container('docker'){
          withCredentials([[
              $class: 'AmazonWebServicesCredentialsBinding',
              credentialsId: 'aws-ecr-staging',
          ]]){
              sh """#!/bin/bash
              echo 'Tagging docker image as ready for production.  For now, this stage of the pipeline does nothing.'
              # docker build -t $PROD_IMAGE .
              # docker push $PROD_IMAGE
              """
          }
        } // container('docker')
      } //steps
    } // stage('Deploy')
  } // stages

  post {
    // only clean up if the build was successful; this allows us to debug failed builds
    success {
        // sh """git clean -fdx"""
        slackSend channel: "#nuscenes-ci", token: "bWyF0sJAVlMPOTs2lUTt5c2N", color: "#00cc00", message: """Success ${env.JOB_NAME} #${env.BUILD_NUMBER} [${env.CHANGE_AUTHOR}] (<${env.BUILD_URL}|Open>)
${env.CHANGE_BRANCH}: ${env.CHANGE_TITLE}"""
    }
    aborted {
        slackSend channel: "#nuscenes-ci", token: "bWyF0sJAVlMPOTs2lUTt5c2N", color: "#edb612", message: """Aborted ${env.JOB_NAME} #${env.BUILD_NUMBER} [${env.CHANGE_AUTHOR}] (<${env.BUILD_URL}|Open>)
${env.CHANGE_BRANCH}: ${env.CHANGE_TITLE}"""
    }
    failure {
        slackSend channel: "#nuscenes-ci", token: "bWyF0sJAVlMPOTs2lUTt5c2N", color: "#c61515", message: """Failed ${env.JOB_NAME} #${env.BUILD_NUMBER} [${env.CHANGE_AUTHOR}] (<${env.BUILD_URL}|Open>)
${env.CHANGE_BRANCH}: ${env.CHANGE_TITLE}"""
    }
    //changed {
        // only run if the current Pipeline run has a different status from previously completed Pipeline
    //}
  } // post

} // Pipeline
