"""
The Astra Client takes customer credentials and performs basic
CRUD operations with the Astra API.
"""
import base64

import requests

from .exceptions import AstraAuthenticationException, AstraException
from .models import AstraUser, AstraUserIntent


class AstraHttpRequester(object):
    """
    Class for requesting the Astra API.
    """
    def __init__(self, api_base):
        self.headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        self.version = "/v1"
        self.base_url = api_base + self.version

    def get(self, url, headers=None):
        """
        Basic implementation of a GET request to the Astra API.
        """
        url = self.base_url + url
        if headers is not None:
            self.headers.update(headers)
        response = requests.get(url, headers=self.headers)
        return response.json()

    def post(self, url, data, headers=None):
        """
        Basic implementation of a POST request to the API.
        """
        url = self.base_url + url
        if headers is not None:
            self.headers.update(headers)
        response = requests.post(url, data, headers=self.headers)
        return response.json()


class AstraBasicAuthRequester(AstraHttpRequester):
    """
    Class for making requests to Astra endpoints that require Basic Authorization.
    """
    def __init__(self, token, *args):
        super(AstraBasicAuthRequester, self).__init__(token, *args)
        self.headers.update({
            "Authorization": "Basic %s" % token,
        })


class AstraOAuthRequester(AstraHttpRequester):
    """
    Class for initiating OAuth requests to the Astra API with a user's access token.
    """
    def __init__(self, access_token, *args):
        super(AstraOAuthRequester, self).__init__(access_token, *args)
        self.headers.update({
            "Authorization": "Bearer %s" % access_token,
        })


class Astra(object):
    """
    The main Astra implementation.

    Imported into client applications like:

    from astra import Astra
    astra_client = Astra(my_client_id, my_client_secret)
    astra_user_info = astra_client.retrieve_user(some_user_id)
    etc.
    """
    def __init__(self, client_id=None, client_secret=None, is_sandbox=True, test_mode=False):
        if not client_id:
            raise AstraAuthenticationException("Missing client_id.")
        if not client_secret:
            raise AstraAuthenticationException("Missing client_secret")

        if is_sandbox and test_mode:
            raise AstraException("is_sandbox and test_mode cannot both be True.")

        basic_auth_token = base64.b64encode("%s:%s" % (client_id, client_secret))
        self.basic_authorization_header = "Bearer %s" % basic_auth_token

        if test_mode:
            self.api_base = ""
        elif is_sandbox:
            self.api_base = "https://api-sandbox.astra.finance"
        else:
            self.api_base = "https://api.astra.finance"

        self.basic_auth_requester = AstraBasicAuthRequester(self.basic_authorization_header, self.api_base)

    def retrieve_access_token(self, authorization_code, redirect_uri):
        # type: (str, str) -> str
        """
        Retrieve an access token from a customer's Authorization code obtained during OAuth with Astra.

        See https://docs.astra.finance/#authenticate-a-user

        Example:

        astra = Astra(my_client_id, my_client_secret)
        code = get_authorization_code_from_frontend()
        access_token = astra.retrieve_access_token(code, "https://example.com/login-success")
        """
        url = "/oauth/token"
        data = {
            "grant_type": "authorization_code",
            "code": authorization_code,
            "redirect_uri": redirect_uri
        }
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        response = AstraBasicAuthRequester(
            self.basic_authorization_header).post(url, data, headers=headers)
        return response

    def refresh_access_token(self, refresh_token, redirect_uri):
        # type: (str, str) -> str
        """
        Refresh an access token for a customer using the customer's
        stored refresh token.

        See https://docs.astra.finance/#refresh-authentication

        Example:

        astra = Astra(my_client_id, my_client_secret)
        refresh_token = some_user.astra_refresh_token
        access_token = astra.refresh_access_token(
            refresh_token,
            "https://example.com/redirect-uri"
        )
        user.access_token = access_token
        user.save()
        """
        url = "/oauth/token"
        data = {
            "grant_type": "authorization_code",
            "refresh_token": refresh_token,
            "redirect_uri": redirect_uri
        }
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        response = self.basic_auth_requester.post(url, data, headers=headers)
        return response

    def retrieve_user(self, user_access_token):
        # type: (str) -> AstraUser
        """
        Retrieve information about a user from their access token.

        Example:

        access_token = some_user.astra_access_token
        astra = Astra(client_id, client_secret)
        astra_user_data = astra.retrieve_user(user_access_token)
        user_status = astra_user_data.status
        etc.
        """
        url = "/user"
        response = AstraOAuthRequester(user_access_token, self.api_base).get(url)
        return AstraUser(**response)

    def create_user_intent(self, data):
        # type: (dict) -> AstraUserIntent
        """
        Retrieve information about a user from their access token.

        Example:

        access_token = some_user.astra_access_token
        astra = Astra(client_id, client_secret)
        user_intent_data = {
            "email": "sir.edmund.hillary@gmail.com",
            "phone": "+16463439898",
            "first_name": "Edmund",
            "last_name": "Hillary",
            "address1": "123 Astra Ave",
            "address2": "Apt 456",
            "city": "Palo Alto"
            "state": "CA",
            "postal_code": "94304",
            "date_of_birth": "1919-07-20",
            "ssn": "9999",
            "ip_address": "79.245.299.113",
        }
        astra_user_intent_data = astra.create_user_intent(user_intent_data)
        astra_user_intent_status = astra_user_intent_data.status
        etc.
        """
        url = "/user_intent"
        response = AstraBasicAuthRequester(self.basic_authorization_header).post(url, data)
        return AstraUserIntent(**response)
