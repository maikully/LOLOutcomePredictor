# Required imports
import requests
import json
import time
import pandas as pd
from tqdm import tqdm


api_key = "RGAPI-a6aec52c-b788-48fd-a704-27b40a7ad266"
region = "NA1"
region5 = "americas"


# get one page of summoner ids for a certain rank
def summ_ID_puller(division, tier, page, summID_list):
    print("pulling division " + str(division) +
          " " + str(tier) + " page " + str(page))
    url_pull = "https://{}.api.riotgames.com/lol/league/v4/entries/RANKED_SOLO_5x5/{}/{}?page={}&api_key={}".format(
        region, division, tier, page, api_key)
    profile_list = requests.get(url_pull).json()
    num_profiles = len(profile_list)
    for profile in range(0, num_profiles):
        summID_list.append(profile_list[profile]['summonerId'])


# get puuid from a summoner id
def puuid_puller(summID, puuid_list):
    url_acct_pull = "https://{}.api.riotgames.com/lol/summoner/v4/summoners/{}?api_key={}".format(
        region, summID, api_key)
    account_info = requests.get(url_acct_pull).json()
    puuid_list.append(account_info["puuid"])


# get 5 last matches from puuid
def match_ID_puller(puuid, matchID_list, pull_errors):
    url_match_pull = "https://{}.api.riotgames.com/lol/match/v5/matches/by-puuid/{}/ids?queue=420&start=0&count=5&api_key={}".format(
        region5, puuid, api_key)
    match_history = requests.get(url_match_pull).json()
    for match in match_history:
        try:
            matchID_list.append(match)
        except KeyError:
            print(match_history)
            print("KeyError occured with account:", puuid)
            pull_errors.append(match_history)


# get the champions and outcome of a match
def match_champs_outcome_puller(match_id, champs_list, outcomes_list):
    url_match_pull = "https://{}.api.riotgames.com/lol/match/v5/matches/{}?api_key={}".format(
        region, match_id, api_key)
    match = requests.get(url_match_pull).json()
    participants = match["info"]["participants"]
    """
    champs_list.append("")
    for x in participants:
        #print(x)
        champs_list[-1] += str(x["championId"]) + ","
    """

    champs_list.append([])
    for x in participants:
        # get champ and mastery
        "https://na1.api.riotgames.com/lol/champion-mastery/v4/champion-masteries/by-summoner/ZrttdtYuZH6_UZ2MiHBv5il0j8M3YaWBozNZE1-BifY6HsrumNv8VZjjpQ/by-champion/17?api_key=RGAPI-a6aec52c-b788-48fd-a704-27b40a7ad266"
        url_match_pull = "https://{}.api.riotgames.com/lol/match/v5/matches/{}?api_key={}".format(
            region5, match_id, api_key)

        # print(x)
        champs_list[-1].append(x["championId"]) 
    teams = match["info"]["teams"]
    if teams[0]["win"]:
        outcomes_list.append([1, 0])
    else:
        outcomes_list.append([0, 1])


def main():
    """
    Get the champions picked and game outcome of the 5 most recent 
    ranked solo/duo 5x5 matches from 12,000 diamond-tier players
    """
    summID_list = []
    puuid_list = []
    matchID_list = []
    pull_errors = []
    outcomes_list = []
    champs_list = []
    # data: https://ddragon.leagueoflegends.com/cdn/12.21.1/data/en_US/champion.json
    """
    # get diamond summoners
    for tier in ["I", "II", "III", "IV"]:
        for page in range(1, 20):
            time.sleep(1.3)
            summ_ID_puller("DIAMOND", tier, page, summID_list)
    df = pd.DataFrame(summID_list, columns=["SummonerID"])
    df.to_csv('data/summID.csv', mode='a')
    print("done pulling summoners!")
            """
    summoner_IDs = pd.read_csv("data/summID.csv")
    summID_list = summoner_IDs["SummonerID"]
    # turn summoner ids into puuids
    """
    for summID_idx in tqdm(range(0, 12000)):
        time.sleep(1)
        if summID_list[summID_idx] == "SummonerID":
            pass

        else:
            try:
                puuid_puller(summID_list[summID_idx], puuid_list)
            except KeyError:
                print("keyerror")
        if summID_idx % 10 == 0:
            df = pd.DataFrame(puuid_list, columns=["PUUID"])
            h = summID_idx == 0
            df.to_csv('data/puuid.csv', mode='a', columns=[
                      "PUUID"], index=False, header=h)
            puuid_list.clear()
    """
    # print(accountID_list)
    #df = pd.DataFrame(accountID_list, columns=["AccountId"])
    #df.to_csv('data/accountId.csv', mode='a')
    #
    print("Done pulling accounts!")

    puuids = pd.read_csv("data/puuid.csv")
    puuid_list = puuids["PUUID"]
    """
    # get 5 last matches for each puuid
    for puuid in tqdm(puuid_list):
        time.sleep(1)
        if puuid == "PUUID":
            pass
        else:
            match_ID_puller(puuid, matchID_list, pull_errors)

    df = pd.DataFrame(matchID_list, columns=["MatchID"])
    df.to_csv('data/MatchID.csv', mode='a', index=False)
    """
    print("Done pulling matchIDs!")

    matches = pd.read_csv("data/MatchID.csv")
    matchID_list = matches["MatchID"]

    champs_list = []
    outcomes_list = []
    train_size = 35000
    # get champions and outcome for each match
    for i, match_id in enumerate(tqdm(matchID_list)):
        #  last entry minus 1 or 2
        if i > 2621 and i < 35000:
            time.sleep(.5)
            if match_id == "MatchId":
                pass
            else:
                match_champs_outcome_puller(
                    match_id, champs_list, outcomes_list)
                if i % 20 == 0:
                    # train data
                    if i < train_size:
                        path1 = "data/Champs_train.csv"
                        path2 = "data/Outcomes_train.csv"
                    # test
                    #  data
                    else:
                        path1 = "data/Champs_test.csv"
                        path2 = "data/Outcomes_test.csv"
                    if i == 0 or i == train_size:
                        h = True
                    else:
                        h = False
                    df = pd.DataFrame(champs_list, columns=[
                        "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"])
                    df.to_csv(path1, mode='a', index=False, header=h)
                    df = pd.DataFrame(outcomes_list, columns=["1", "2"])
                    df.to_csv(path2, mode='a', index=False, header=h)
                    champs_list.clear()
                    outcomes_list.clear()
    """
    df = pd.DataFrame(champs_list, columns=["Champs"])
    df.to_csv('data/Champs.csv', mode='a', index=False)
    df = pd.DataFrame(outcomes_list, columns=["Outcome"])
    df.to_csv('data/Outcomes.csv', mode='a', index=False)
    """
    print("Done pulling champs!")
    print("Done pulling outcomes!")
    print("Data scraping complete")


if __name__ == '__main__':
    main()
