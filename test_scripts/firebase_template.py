# Database Class (Using Google Firebase)
class firebase_connector():

    # Init constructor
    def __init__(self):

        # Import Firebase / DB Keys
        import json
        import firebase_admin
        from firebase_admin import credentials
        from firebase_admin import db
        self.db = db

        # Get database config for connection
        with open('../keys/api_info.json') as file:
            self.api_info = json.load(file)

        # Establish connection
        cred = credentials.Certificate('../keys/service_key.json')
        firebase_admin.initialize_app(cred, {
            'databaseURL': self.api_info['databaseURL']
        })

    # Handle data get
    def get_node(self, path):
        ref = self.db.reference(path)
        return ref.get()

    # Handle data post
    def set_node(self, path, data):
        ref = self.db.reference(path)
        ref.set(data)


fc = firebase_connection()
fc.set_node("/", {'hello':'world'})
