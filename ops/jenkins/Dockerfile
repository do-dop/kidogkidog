# Jenkins controller that can run the Ansible deploy pipeline itself.
# Pin the tool versions so every build uses the same Ansible (see study week5/05).
FROM jenkins/jenkins:lts-jdk21

USER root
RUN apt-get update \
 && apt-get install -y --no-install-recommends python3 python3-venv git openssh-client \
 && rm -rf /var/lib/apt/lists/* \
 && python3 -m venv /opt/ansible \
 && /opt/ansible/bin/pip install --no-cache-dir "ansible-core==2.19.*" "ansible-lint==25.*" \
 && git config --system --add safe.directory '*'
ENV PATH="/opt/ansible/bin:${PATH}"

USER jenkins
COPY plugins.txt /usr/share/jenkins/ref/plugins.txt
RUN jenkins-plugin-cli --plugin-file /usr/share/jenkins/ref/plugins.txt
ENV CASC_JENKINS_CONFIG=/casc/jenkins.yaml
ENV JAVA_OPTS="-Djenkins.install.runSetupWizard=false"
