"""
Module for models of Astra objects
TODO: Will this instantiation work with the JSON response from `requests`?
TODO: Add validation and helpful errors in the event of invalid data.
"""


class AstraUserIntent(object):
    """
    Model for an Astra UserIntent object
    """
    def __init__(self, **kwargs):
        # TODO: Will this instantiation work with the JSON response?
        self.id = kwargs["id"]
        self.email = kwargs["email"]
        self.phone = kwargs["phone"]
        self.first_name = kwargs["first_name"]
        self.last_name = kwargs["last_name"]
        self.oauth_token_issued = kwargs["oauth_token_issued"]
        self.status = kwargs["status"]


class AstraUser(object):
    """Model for an Astra User object"""
    def __init__(self, **kwargs):
        # TODO: Will this instantiation work with the JSON response?
        self.id = kwargs['id']
        self.email = kwargs['email']
        self.phone = kwargs['phone']
        self.first_name = kwargs['first_name']
        self.last_name = kwargs['last_name']
        self.status = kwargs['status']
