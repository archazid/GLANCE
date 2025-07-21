# -*-coding:utf-8-*-

import sys

# --------------------------------
# Code Path
# --------------------------------
CODE_PATH = "C:/Users/Archazid/projects/GLANCE/"
sys.path.append(CODE_PATH)

# --------------------------------
# Testing Mode
# --------------------------------
# Set to True to run a smaller, faster version of the experiments for debugging.
# This will limit the number of releases processed.
IS_TEST_RUN: bool = False

# --------------------------------
# Cached Result
# --------------------------------
# Set to True to load cached results instead of re-computing them.
# This is useful for speeding up subsequent runs.
USE_CACHE: bool = False

# --------------------------------
# Dataset Project
# --------------------------------
# Full list of 142 project releases, sorted chronologically for each project.
ALL_PROJECT_RELEASES = [
    # Ambari
    "ambari-1.2.0",
    "ambari-2.1.0",
    "ambari-2.2.0",
    "ambari-2.4.0",
    "ambari-2.5.0",
    "ambari-2.6.0",
    "ambari-2.7.0",
    # ActiveMQ (amq)
    "amq-5.0.0",
    "amq-5.1.0",
    "amq-5.2.0",
    "amq-5.4.0",
    "amq-5.5.0",
    "amq-5.6.0",
    "amq-5.7.0",
    "amq-5.8.0",
    "amq-5.9.0",
    "amq-5.10.0",
    "amq-5.11.0",
    "amq-5.12.0",
    "amq-5.14.0",
    "amq-5.15.0",
    # Bookkeeper
    "bookkeeper-4.0.0",
    "bookkeeper-4.2.0",
    "bookkeeper-4.4.0",
    # Calcite
    "calcite-1.6.0",
    "calcite-1.8.0",
    "calcite-1.11.0",
    "calcite-1.13.0",
    "calcite-1.15.0",
    "calcite-1.16.0",
    "calcite-1.17.0",
    "calcite-1.18.0",
    # Cassandra
    "cassandra-0.7.4",
    "cassandra-0.8.6",
    "cassandra-1.0.9",
    "cassandra-1.1.6",
    "cassandra-1.1.11",
    "cassandra-1.2.11",
    # Flink
    "flink-1.4.0",
    "flink-1.6.0",
    # Groovy
    "groovy-1.0",
    "groovy-1.5.5",
    "groovy-1.6.0",
    "groovy-1.7.3",
    "groovy-1.7.6",
    "groovy-1.8.1",
    "groovy-1.8.7",
    "groovy-2.1.0",
    "groovy-2.1.6",
    "groovy-2.4.4",
    "groovy-2.4.6",
    "groovy-2.4.8",
    "groovy-2.5.0",
    "groovy-2.5.5",
    # HBase
    "hbase-0.94.1",
    "hbase-0.94.5",
    "hbase-0.98.0",
    "hbase-0.98.5",
    "hbase-0.98.11",
    # Hive
    "hive-0.14.0",
    "hive-1.2.0",
    "hive-2.0.0",
    "hive-2.1.0",
    # Ignite
    "ignite-1.0.0",
    "ignite-1.4.0",
    "ignite-1.6.0",
    # Log4j2
    "log4j2-2.0",
    "log4j2-2.1",
    "log4j2-2.2",
    "log4j2-2.3",
    "log4j2-2.4",
    "log4j2-2.5",
    "log4j2-2.6",
    "log4j2-2.7",
    "log4j2-2.8",
    "log4j2-2.9",
    "log4j2-2.10",
    # Mahout
    "mahout-0.3",
    "mahout-0.4",
    "mahout-0.5",
    "mahout-0.6",
    "mahout-0.7",
    "mahout-0.8",
    # Maven (mng)
    "mng-3.0.0",
    "mng-3.1.0",
    "mng-3.2.0",
    "mng-3.3.0",
    "mng-3.5.0",
    "mng-3.6.0",
    # Nifi
    "nifi-0.4.0",
    "nifi-1.2.0",
    "nifi-1.5.0",
    "nifi-1.8.0",
    # Nutch
    "nutch-1.1",
    "nutch-1.3",
    "nutch-1.4",
    "nutch-1.5",
    "nutch-1.6",
    "nutch-1.7",
    "nutch-1.8",
    "nutch-1.9",
    "nutch-1.10",
    "nutch-1.12",
    "nutch-1.13",
    "nutch-1.14",
    "nutch-1.15",
    # Storm
    "storm-0.9.0",
    "storm-0.9.3",
    "storm-1.0.0",
    "storm-1.0.3",
    "storm-1.0.5",
    # Tika
    "tika-0.7",
    "tika-0.8",
    "tika-0.9",
    "tika-0.10",
    "tika-1.1",
    "tika-1.3",
    "tika-1.5",
    "tika-1.7",
    "tika-1.10",
    "tika-1.13",
    "tika-1.15",
    "tika-1.17",
    # Struts (ww)
    "ww-2.0.0",
    "ww-2.0.5",
    "ww-2.0.10",
    "ww-2.1.1",
    "ww-2.1.3",
    "ww-2.1.7",
    "ww-2.2.0",
    "ww-2.2.2",
    "ww-2.3.1",
    "ww-2.3.4",
    "ww-2.3.10",
    "ww-2.3.15",
    "ww-2.3.17",
    "ww-2.3.20",
    "ww-2.3.24",
    # Zookeeper
    "zookeeper-3.4.6",
    "zookeeper-3.5.1",
    "zookeeper-3.5.2",
    "zookeeper-3.5.3",
]

# A smaller subset of releases for quick testing and debugging.
TEST_PROJECT_RELEASES = ["ambari-1.2.0", "ambari-2.1.0", "ambari-2.2.0", "ambari-2.4.0"]


# ==============================================================================
#  DO NOT CHANGE THIS SECTION.
# ==============================================================================
# The final configuration used by the application, determined by the settings above.
if IS_TEST_RUN:
    PROJECT_RELEASE_LIST = TEST_PROJECT_RELEASES
else:
    PROJECT_RELEASE_LIST = ALL_PROJECT_RELEASES
