# Required imports
import requests
import json
import time
import pandas as pd




# Information for Riot API
api_key = "RGAPI-a6aec52c-b788-48fd-a704-27b40a7ad266"
region = "NA1"
from tqdm import tqdm

# Code for pulling list of summoner IDs

# Function to pull a single page of summoner IDs for given rank and tier
def summ_ID_puller(division, tier, page, summID_list):
    print("pulling division " + str(division) + " " + str(tier) + " page " + str(page))
    url_pull = "https://{}.api.riotgames.com/lol/league/v4/entries/RANKED_SOLO_5x5/{}/{}?page={}&api_key={}".format(
        region, division, tier, page, api_key)
    profile_list = requests.get(url_pull).json()
    num_profiles = len(profile_list)

    for profile in range(0, num_profiles):
        summID_list.append(profile_list[profile]['summonerId'])



# Function to get the encrypted account ID from summoner ID
def acct_ID_puller(summID, accountID_list):
    url_acct_pull = "https://{}.api.riotgames.com/lol/summoner/v4/summoners/{}?api_key={}".format(
        region, summID, api_key)
    account_info = requests.get(url_acct_pull).json()
    accountID_list.append(account_info["accountId"])
    #print(account_info["accountId"])


# Function to pull the 5 most recent matches for a given account ID
def match_ID_puller(acctid, matchID_list, pull_errors):
    url_match_pull = "https://{}.api.riotgames.com/lol/match/v4/matchlists/by-account/{}?queue=420&api_key={}".format(
        region, acctid, api_key)
    match_history = requests.get(url_match_pull).json()
    for i in range(0, 5):
        try:
            match_id = match_history['matches'][i]['gameId']
            matchID_list.append(match_id)

        except KeyError:
            print(match_history)
            print("KeyError occured with account:", acctid)
            pull_errors.append(match_history)


# Function to pull the 5 most recent matches for a given account ID
def match_champs_outcome_puller(match_id, champs_list, outcomes_list):
    url_match_pull = "https://{}.api.riotgames.com/lol/match/v5/matches/{}?api_key={}".format(
        region, match_id, api_key)
    match = requests.get(url_match_pull).json()
    participants = match["info"]["participants"]
    champs_list.append([])
    for x in participants:
        champs_list[-1].append(x["championID"])
    teams = match["info"]["teams"]
    if teams[0]["win"]:
        outcomes_list.append([1, 0])
    else:
        outcomes_list.append([0, 1])



def main():

    summID_list = []
    accountID_list = []
    matchID_list = []
    pull_errors = []
    outcomes_list = []
    champs_list = []
    """
    # get diamond summoner
    for tier in ["I", "II", "III", "IV"]:
        for page in range(1, 20):
            time.sleep(1.3)
            summ_ID_puller("DIAMOND", tier, page, summID_list)
    df = pd.DataFrame(summID_list, columns=["Summoner ID"])
    df.to_csv('summID.csv', mode='a')
    print("done pulling summoners!")
            """
    summoner_IDs = pd.read_csv("summID.csv")
    summID_list=summoner_IDs["Summoner ID"]

    # Code for pulling list of account IDs from summoner IDs
    for summID_idx in tqdm(range(0, 12000)):
        time.sleep(1)
        if summID_list[summID_idx] == "Summoner ID":
            pass

        else:
            try:
                acct_ID_puller(summID_list[summID_idx], accountID_list)
            except KeyError:
                print("keyerror")


    #print(accountID_list)
    df=pd.DataFrame(accountID_list, columns=["AccountId"])
    df.to_csv('accountId.csv', mode='a')
    print("Done pulling accounts!")

    # Step 3: Pulling 5 most recent matches for each player

    for acct_id in tqdm(accountID_list):
        time.sleep(1)
        if acct_id == "AccountId":
            pass
        else:
            match_ID_puller(acct_id, matchID_list, pull_errors)

    df=pd.DataFrame(matchID_list, columns=["MatchId"])
    df.to_csv('MatchId.csv', mode='a')
    print("Done pulling matchIDs!")

    champs_list=[]
    outcomes_list=[]

    for match_id in tqdm(matchID_list):
        time.sleep(1)
        if match_id == "MatchId":
            pass
        else:
            match_champs_outcome_puller(match_id, champs_list, outcomes_list)
    print(champs_list)
    print(outcomes_list)

    df=pd.DataFrame(champs_list, columns=["Champs"])
    df.to_csv('Champs.csv', mode='a')
    print("Done pulling champs!")

    df=pd.DataFrame(outcomes_list, columns=["Outcome"])
    df.to_csv('Outcomes.csv', mode='a')
    print("Done pulling outcomes!")

if __name__ == '__main__':
    main()
