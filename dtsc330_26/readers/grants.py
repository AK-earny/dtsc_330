# CamelCase
# snake_case -- this is what python programmers use
import pandas as pd
import sqlalchemy


class Grants:  # class names in python are camel case (e.g. GrantReader)
    def __init__(self, path: str | None = None):
        """Create and parse a Grants file

        Args:
            path (str): the location of the file on the disk
                If empty, it defaults to pulling from the database
        """
        # What is self?
        # "Self is the specific instance of the object" - Computer Scientist
        # Store shared variables in self
        if path is None:
            self.df, self.grantee_df = self._from_db()
        self.path = path
        self.df, self.grantee_df = self._parse(path)

    def _parse(self, path: str):
        """Parse a grants file"""
        df = pd.read_csv(path, compression="zip")

        mapper = {
            "APPLICATION_ID": "application_id",  # _id means an id
            "BUDGET_START": "start_at",  #  _at means a date
            "ACTIVITY": "grant_type",
            "TOTAL_COST": "total_cost",
            "PI_NAMEs": "pi_names",  # you will notice, homework references this
            "ORG_NAME": "organization",
            "ORG_CITY": "city",
            "ORG_STATE": "state",
            "ORG_COUNTRY": "country",
        }
        # make column names lowercase
        # maybe combine for budget duration?
        df = df.rename(columns=mapper)[mapper.values()]

        # Added after homework
        # ====================
        df["affiliation"] = df.apply(
            lambda row: ", ".join(
                [
                    v
                    for v in [
                        row["organization"],
                        row["city"],
                        row["state"],
                        row["country"],
                    ]
                    if not pd.isna(v)
                ]
            ),
            axis=1,
        ).str.lower()

        grantees = df[["application_id", "pi_names", "affiliation"]].dropna(how="any")
        grantees["pi_name"] = grantees["pi_names"].str.split(";")
        grantees = grantees.explode("pi_name").reset_index(drop=True)

        grantees["pi_name"] = (
            grantees["pi_name"].str.lower().str.replace("(contact)", "").str.strip()
        )
        names = grantees["pi_name"].apply(lambda x: x.split(","))
        grantees["surname"] = names.apply(lambda x: x[0]).str.strip()
        grantees["forename"] = (
            names.apply(lambda x: x[1]).str.replace(".", "").str.strip()
        )
        grantees["initials"] = grantees["forename"].apply(
            lambda x: [v[0] for v in x.split(" ") if len(v) > 0]
        )
        # ====================

        return (
            df.drop(columns=["pi_names"]),
            grantees[["surname", "forename", "initials", "affiliation"]],
        )

    def to_db(self, path: str = "C:/Users/alexk/Downloads/RePORTER_PRJ_C_FY2025.zip"):
        """Send the read-in data to the database

        Args:
            path (str, optional): Location of sqlite file.
                Defaults to 'data/article_grant_db.sqlite'.
        """
        # Define the connection
        engine = sqlalchemy.create_engine("sqlite:///data/article_grant_db.sqlite")
        connection = engine.connect()

        # Always append. Deletion should be more thoughtful
        # NEVER alter raw data.
        # Pandas has its own index. That is different from the primary key.
        # If you want, you can use the primary key as an index. I don't.
        # It's complicated.

        self.df[["application_id", "start_at", "grant_type", "total_cost"]].to_sql(
            "grants", connection, if_exists="append", index=False
        )

    def _from_db(self):
        """Load the data from the database"""
        engine = sqlalchemy.create_engine("sqlite:///data/article_grant_db.sqlite")
        connection = engine.connect()
        df = pd.read_sql("SELECT * FROM grants", connection)
        return df

    def get_grants(self):
        """Get parsed grants"""
        return self.df

    # Added after homework
    # ====================
    def get_grantees(self):
        """Get parsed grantees"""
        return self.grantee_df.rename(
            {
                "LastName": "surname",
                "ForeName": "forename",
                "Initials": "initials",
                "Affiliation": "affiliation",
            }
        )


if __name__ == "__main__":
    # This is for debugging
    grants = Grants("C:/Users/alexk/Downloads/RePORTER_PRJ_C_FY2025.zip")
    grants.to_db()