# my-little-elephant/src/elephant/messaging/nextcloud_talk.py

import logging
import base64
from typing import Optional, List
import httpx # httpx is a good choice for async HTTP requests
import datetime

# Ensure the project root is in sys.path for internal imports
import sys
import os

# Adjust this path based on your actual project root.
# If this file is in 'my-little-elephant/src/elephant/messaging/',
# then 'src/elephant' needs to be added to sys.path.
# This assumes the script is run from the 'my-little-elephant' root.
project_root = os.path.abspath(os.getcwd())
elephant_src_path = os.path.join(project_root, 'src', 'elephant')

# Add the 'src/elephant' directory to sys.path if it's not already there.
# This should ideally be handled at the application's entry point,
# but included here for standalone testing within JupyterLab.
if elephant_src_path not in sys.path:
    sys.path.insert(0, elephant_src_path)

# Now, import from the project's base messaging module
from messaging.base import MessagingClient, IncomingMessage, SendResult

logger = logging.getLogger(__name__)

class NextcloudTalkClient(MessagingClient):
    """
    A messaging client for Nextcloud Talk.
    """
