FROM cern/cc7-base
EXPOSE 8083

# ---------------------------
# 1. Update EPEL Repository URLs
# ---------------------------
RUN sed -i 's#http://linuxsoft.cern.ch/epel/7/#http://linuxsoft.cern.ch/internal/archive/epel/7/#g' /etc/yum.repos.d/epel.repo

# ---------------------------
# 2. Install System Dependencies
# ---------------------------
RUN yum update -y && yum install -y \
      ImageMagick \
      httpd \
      npm \
      php \
      python3-pip \
      wget \
      bzip2 \
      epel-release \
      root \
      python3-root \
    && yum clean all

# ---------------------------
# 3. Upgrade pip to a Compatible Version
# ---------------------------
RUN pip3 install --upgrade pip==20.3.4

# ---------------------------
# 4. Set Python Alias (Optional)
# ---------------------------
RUN echo "alias python=python3" >> ~/.bashrc

# ---------------------------
# 5. Create Application Directory
# ---------------------------
RUN mkdir -p /app
WORKDIR /app

# ---------------------------
# 6. Copy and Install Python Requirements
# ---------------------------
COPY requirements.txt /app/requirements.txt

# Install Python dependencies (only once to avoid duplication)
RUN pip3 install --no-cache-dir -r requirements.txt

# ---------------------------
# 7. Copy Application Code into Docker Image
# ---------------------------
COPY . /app

# ---------------------------
# 8. Set Proper Permissions for Application Directory
# ---------------------------
RUN chmod -R 755 /app

# ---------------------------
# 9. Set Environment Variables
# ---------------------------
ENV HOME /root
ENV REQUESTS_CA_BUNDLE /etc/ssl/certs/ca-bundle.crt
ENV ADQM_SSLCERT /etc/robots/robotcert.pem
ENV ADQM_SSLKEY /etc/robots/robotkey.pem

# Create additional directories and set environment variables
RUN mkdir -p /var/adqm
ENV ADQM_TMP /var/adqm
ENV ADQM_DB /db/
ENV ADQM_PUBLIC /var/www/
ENV ADQM_CONFIG /var/www/public/config/
ENV ADQM_PLUGINS /var/www/cgi-bin/plugins/
ENV ADQM_MODELS /var/www/cgi-bin/models/
# ENV ADQM_MODULES /var/www/cgi-bin/modules/

# ---------------------------
# 10. Set Up the Web Application
# ---------------------------
WORKDIR /webapp
COPY webapp/package.json /webapp/package.json
RUN npm install

COPY webapp /webapp
RUN npm run build

# Copy build artifacts to public directories
RUN cp -r /webapp/build /var/www/public
RUN cp -r /webapp/build /webapp/public

# ---------------------------
# 11. Create and Set Permissions for Results Directories
# ---------------------------
RUN mkdir -p /var/www/results/pdfs /var/www/results/pngs /var/www/results/jsons
RUN chmod -R 777 /var/www/results /var/www/results/pdfs /var/www/results/pngs /var/www/results/jsons

# ---------------------------
# 12. Configure Apache
# ---------------------------
COPY httpd.conf /etc/httpd/conf/httpd.conf

# ---------------------------
# 13. Copy Application Code for CGI
# ---------------------------
COPY index.py /var/www/cgi-bin/index.py
COPY autodqm /var/www/cgi-bin/autodqm
COPY autoref /var/www/cgi-bin/autoref
COPY plugins /var/www/cgi-bin/plugins
COPY models /var/www/cgi-bin/models/
# COPY modules /var/www/cgi-bin/modules
COPY config /var/www/public/config

# ---------------------------
# 14. Adjust Group Permissions
# ---------------------------
RUN chgrp -R 1000 /run && chmod -R g=u /run
RUN chgrp -R 1000 /etc/httpd/logs && chmod -R g=u /etc/httpd/logs

# ---------------------------
# 15. Create Necessary Directories and Set Permissions
# ---------------------------
RUN mkdir /db /run/secrets
RUN chown -R 1000:1000 /db /var/www /run/secrets
RUN chmod -R 755 /db /var/www/cgi-bin/models

# ---------------------------
# 16. Create Symbolic Links for Logs
# ---------------------------
RUN ln -s /dev/stdout /home/access_log
RUN ln -s /dev/stderr /home/error_log

# ---------------------------
# 17. Set Permissions for Log Files
# ---------------------------
RUN chown 1000:1000 /home/error_log
RUN chown 1000:1000 /home/access_log   
RUN chmod 755 /home/error_log
RUN chmod 755 /home/access_log  

# ---------------------------
# 18. (Optional) Create and Switch to Non-Root User for Security
# ---------------------------
# Uncomment the following lines to run the application as a non-root user
# RUN useradd -m appuser
# RUN chown -R appuser:appuser /app
# USER appuser

# ---------------------------
# 19. Final Command to Run Apache in the Foreground
# ---------------------------
CMD ["/usr/sbin/httpd","-D","FOREGROUND"]

