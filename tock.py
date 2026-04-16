# ================================ Imports ============================================
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import Circle
import matplotlib.patches as patches
from dataclasses import dataclass
from typing import List
from itertools import product
from random import choice


# ====================== Definition of constants ======================================
PLACES_PER_SEGMENT = 16  # DO NOT CHANGE!!
RANKS = '2 3 4 5 6 7 8 9 10 J Q K A'.split()
SUITS = '♣ ♢ ♡ ♠ ♡ ♠'.split()
DEAL_ORDER = [5, 4, 4]
AVERAGE_GAME_LENGTH = 35

# ================================= Classes ===========================================
# class Deck:
#     """"create a deck of 13 * nplayers unique cards"""
#
#     def __init__(self, nplayers):
#         self.cards = [i for i in range(13 * nplayers)] 
#         self.rng = np.random.default_rng()
#         self.carditer = self.getcard()      
#
#     def getcard(self):
#         """iterator return a card from a shuffled deck. If deck is empty deck is reshuffled"""
#         while True:
#             self.rng.shuffle(self.cards)
#             for card in self.cards:
#                 yield card
#
#     def deal(self, ncards):                
#         """returns ncards from a shuffled deck. If deck is empty deck is reshuffled"""
#         return [next(self.carditer) for _ in range(ncards)]
#
class Deck:
    def __init__(self, nplayers):
        self.nplayers = nplayers
        self.cards = list(range(13 * nplayers))
        self.rng = np.random.default_rng()
        self.pointer = 0
        self._shuffle()

    def _shuffle(self):
        self.rng.shuffle(self.cards)
        self.pointer = 0

    def deal(self, ncards):
        hand = []
        for _ in range(ncards):
            # If we reached the end of the cards, reshuffle
            if self.pointer >= len(self.cards):
                self._shuffle()
            
            hand.append(self.cards[self.pointer])
            self.pointer += 1
        return hand


@ dataclass
class Player:
    """holds the state of the player"""
    offset: int  # absolute postion on board of the startfield of the player
    locs: List[int]  # relative location of the player's pawns (4 in total)
    cards: List[int]  # cards of the player

    def __repr__(self):
        try:
            return f'offset:{self.offset} | locs:{self.locs} | cards:{self.cards} | {[SUITS[card//13]+RANKS[card%13] for card in self.cards if card != -1]}'       
        except:
            breakpoint()


class Board:
    """holds the board for rendering the game"""

    COLORS = ['green', 'yellow', 'blue', 'red', 'purple', 'black']

    def __init__(self, players, current_player, round) -> None:       
        self.nplayers = len(players)
        self.nfields = self.nplayers * PLACES_PER_SEGMENT   
        plt.ion()  # required to force updating the plot
        self.drawboard()
        self.update(players, current_player, round)
        
    def drawboard(self):

        # define constants
        xmin, xmax = -2.5, 1.5
        ymin, ymax = -1.5, 1.5
        scale = 3
        radius=0.03
        
        # setup the figure
        self.fig, self.ax = plt.subplots(figsize=(scale * (xmax - xmin), scale * (ymax - ymin)), facecolor='burlywood', tight_layout=True)
        self.ax.set_xlim(xmin, xmax)
        self.ax.set_ylim(ymin, ymax)
        self.ax.axis('off')

        # create field
        step = 2*np.pi / self.nfields
        self.xfield = [np.cos(np.pi/2 - phi) for phi in np.arange(0, 2*np.pi, step)]
        self.yfield = [np.sin(np.pi/2 - phi) for phi in np.arange(0, 2*np.pi, step)]
        self.fields = [Circle((x, y), radius, fc='w', ec='gray') for (x, y) in zip(self.xfield, self.yfield)]
        
        # create homebase
        self.xhomebase = []
        self.yhomebase = []
        for playernumber, phi in enumerate([i * 2 * np.pi / self.nplayers for i in range(self.nplayers)]):
            xlist = [(1 - (i+1)*radius*3) * np.cos(np.pi/2 - phi) for i in range(4)]
            ylist = [(1 - (i+1)*radius*3) * np.sin(np.pi/2 - phi) for i in range(4)]
            self.xhomebase.append(xlist)
            self.yhomebase.append(ylist)
            for x, y in zip(xlist, ylist):
                self.fields.append(Circle((x, y), radius, fc='w', ec=Board.COLORS[playernumber]))
        
        # create startgrid
        self.xstartgrid = []
        self.ystartgrid = []
        for playernumber, phi in enumerate([i * 2 * np.pi / self.nplayers for i in range(self.nplayers)]):
            xlist = [(1 + (i+1)*radius*3) * np.cos(np.pi/2 - phi - 2*np.pi / self.nfields) for i in range(4)]
            ylist = [(1 + (i+1)*radius*3) * np.sin(np.pi/2 - phi - 2*np.pi / self.nfields) for i in range(4)]
            self.xstartgrid.append(xlist)
            self.ystartgrid.append(ylist)
            for x, y in zip(xlist, ylist):
                self.fields.append(Circle((x, y), radius, fc='w', ec=Board.COLORS[playernumber]))

        # add fields to axis
        for field in self.fields:
            self.ax.add_patch(field)
      
        # create pawns
        self.pawns = [Circle((10,10), radius, fc=color, ec=color) for color in Board.COLORS[:self.nplayers] for i in range(4)]
        for pawn in self.pawns:
            self.ax.add_patch(pawn)
        
        # create textbox for carddisplay
        self.roundtext = self.ax.text(xmin, ymax,
                                      "spam", 
                                      size=15, 
                                      color='k', 
                                      horizontalalignment='left', 
                                      verticalalignment='top')
        self.cardtext = [self.ax.text(xmin, ymax - (i + 1) * 0.1,
                                      "spam", 
                                      size=15, 
                                      color=color, 
                                      horizontalalignment='left', 
                                      verticalalignment='top') for i, color in enumerate(Board.COLORS[:self.nplayers])]

        # load and show image on the board
        image = plt.imread('tockboard.png')
        im = self.ax.imshow(image, extent=(-0.5,0.5,-0.5,0.5))
        patch = patches.Circle((0, 0), radius=0.5, transform=self.ax.transData)
        im.set_clip_path(patch)

        # force drawing the board
        self.fig.canvas.draw()  
        self.fig.canvas.flush_events()         
        
    def index2position(self, index, playernumber, pawnnumber):
        offset = playernumber * PLACES_PER_SEGMENT
        
        # pawn on the field
        if 0 < index <= self.nfields:
            index = (index + offset) % self.nfields
            position = (self.xfield[index], self.yfield[index])
            return position

        # pawn in the homebase
        if index == self.nfields + 1:
            #index = index - self.nfields - 1
            position = self.xhomebase[playernumber][pawnnumber], self.yhomebase[playernumber][pawnnumber]
            return position

        # pawn in the startgrid
        if index == 0:
            position = self.xstartgrid[playernumber][pawnnumber], self.ystartgrid[playernumber][pawnnumber]

        return position

    def update(self, players, current_player, round):

        # draw the pawns        
        for playernumber, player in enumerate(players):   
            for pawnnumber, loc in enumerate(player.locs):
                x, y = self.index2position(loc, playernumber, pawnnumber)
                self.pawns[playernumber*4 + pawnnumber].set_center((x,y))

        # draw the cards
        self.roundtext.set_text(f'ROUND: {round+1:2}')
        for i, (text, player) in enumerate(zip(self.cardtext, players)):
            s = f'>p{i}: ' if player is current_player else f'  p{i}: '
            
            for card in player.cards:
                s += SUITS[card//13] + RANKS[card%13] + ' , '
            text.set_text(s)
            pass   
        
        # force drawing the board
        self.fig.canvas.draw()  
        self.fig.canvas.flush_events()         
        




class TockGame:

    deal_order = DEAL_ORDER  # adds up to 13 always

    def __init__(self, nplayers=6, render=True):
        self.rng = np.random.default_rng()
        self.nplayers = nplayers
        self.nfields = self.nplayers * PLACES_PER_SEGMENT
        self.poslist = [i for i in range(1, self.nfields + 1)] * 3  # used as lookup for position calculations
        self.reset()
        if render:
            self.board = Board(self.players, self.current_player, self.round)
        else:
            self.board = None
    

    def _validate_pawn(self, player, pawn, value, execute=False):
        
        # handle card value of 4
        if value == 4:
            value = -4

        # check if pawn makes it to homebase
        # pawn can not be blocked by other player because value < LENGTH_SEGMENT
        if player.locs[pawn] + value > self.nfields:
            if execute:
                player.locs[pawn] = self.nfields + 1
            return True 

        # check interference with other players pawns

        # first compute absolute positions before and after move
        start = (player.offset + player.locs[pawn]) % self.nfields
        stop = self.poslist[self.nfields + start + value - 1]
        
        # check if move is blocked by other players pawn in startpostion
        for _player in self.players:
            if (_player is not player) and (1 in _player.locs):  # a pawn of another player is on the startposition (=1)
                abspos = _player.offset + 1  # absolute startposition
                # check if abspos is in between pre and post
                b = self.nfields + start - 1
                a = b + value 
                if abspos in self.poslist[a: b : -1 if a > b else +1]: 
                    return False

        # make the move
        if execute:
            # check if another pawn is at the stop position
            for _player in self.players:
                offset = _player.offset
                for ii, pos in enumerate(_player.locs):
                    if stop == self.poslist[self.nfields + offset + pos - 1]:
                        # slaan if pawn is not already in homebase
                        if _player.locs[ii] != self.nfields + 1:
                            _player.locs[ii] = 0
            
            # move the pawn
            player.locs[pawn] = self.poslist[self.nfields + player.locs[pawn] + value - 1]
    
        return True

    def _validate_action(self, player, action, execute=False):
        """validates if action by player is possible"""
        # compute card_value
        card, pawn = action
        card_value = player.cards[card] % 13 + 2  # add 2 to get in the range 2...14

        # check if pawn is in startgrid
        if player.locs[pawn] == 0:
                
            # pawn in start grid
                
            if card_value in [13, 14] :  # heer of aas
                # place pawn on the field if in startgrid
                return self._validate_pawn(player, pawn, 1, execute=execute)

            else:  
                # not a valid move
                return False
            
        # check if pawn is in homebase
        elif player.locs[pawn] > self.nfields:
            return False

        else:
            # pawn on the field
    
            if 2 <= card_value <= 12:
                # move pawn card_value fields
                # boer: move 11; vrouw: move 12
                return self._validate_pawn(player, pawn, card_value, execute=execute)

            elif card_value == 14: # aas
                return self._validate_pawn(player, pawn, 1, execute=execute)

            else:
                # card is an Heer but pawn is not on startgrid
                return False


    def _get_next_player(self, player):
        """returns the next player in the playerslist that has at least one card"""
        oldplayer = player
        while True:
            player = self.players[(self.players.index(player) + 1) % self.nplayers]
            action_space = self._get_actionspace(player)
            if len(action_space) == 0:
                player.cards = []  # remove cards if no actions are possible for player
                self.current_action = None  # this means the current action is none
                if player is oldplayer:  # handle case if all players lost their cards
                    self._next_round()
            else:
                break  # a next player is found
        return player, action_space

    def _get_actionspace(self, player):
        """return a list of all possible actions for player"""
        action_space = [action for action in product(range(len(player.cards)), range(4)) \
                        if self._validate_action(player, action)]        
        return action_space

    def _execute_action(self, player, action):
        """executes action of player"""
        self._validate_action(player, action, execute=True)
        player.cards.remove(player.cards[action[0]])  # remove card
    

    def step(self, action):
        """
        performs action on the current player and returns the gamestate after the action
        if action is not valid, a random action from the possible actions is executed
        """

        # if action is not valid, take a random valid action
        if action not in self.action_space:
            action = choice(self.action_space)    

        # execute the action   
        self.current_action = action 
        self._execute_action(self.current_player, self.current_action)

        # determine the next player 
        self.current_player, self.action_space = self._get_next_player(self.current_player) 

        return self._get_gamestate()


    def render(self):
        """renders the current gamestate"""
        if self.board is not None:
            self.board.update(self.players, self.current_player, self.round)


    def _done(self):
        """determines if the game has finished and if so, which player won"""
        number_of_pawns_in_homebase = [sum([loc==(self.nfields+1) for loc in player.locs]) for player in self.players]
        
        if 4 not in number_of_pawns_in_homebase:
            done, winner = False, -1
        else:
            done, winner = True, number_of_pawns_in_homebase.index(4)
        
        return done, winner


    def _get_gamestate(self):
        rew = None
        done, winner = self._done()
        obs = [player.locs for player in self.players]
        info = {
                'player': self.players.index(self.current_player),
                'cards' : self.current_player.cards,
                'space' : self.action_space,
                'action' : self.current_action,
                'round' : self.round + 1,
                'winner' : winner
                }
        return obs, rew, done, info

    def _next_round(self):
        self.round += 1
        for player in self.players:
            player.cards = self.deck.deal(TockGame.deal_order[self.round%3])

    def reset(self):
        """resets the game to the starting position and returns the gamestate"""

        # reset all players
        self.round = 0
        self.deck = Deck(self.nplayers)
        self.players = [Player(i * PLACES_PER_SEGMENT, [0]*4, self.deck.deal(TockGame.deal_order[self.round%3])) for i in range(self.nplayers)]
        self.current_player, self.action_space = self._get_next_player(self.players[-1])
        self.current_action = False
        
        # return game state
        return self._get_gamestate()

    def __repr__(self) -> str:
        s  = '=======================================\n'
        s += '         Current Game State            \n'
        s += '=======================================\n'
        for id, player in enumerate(self.players):
            s += f'Player:{id} | ' + player.__repr__() + '\n'
        s += '=======================================\n'    
        return s    



# =============================== Functions ======================================

def make(nplayers=6, render=False):
    if nplayers not in [2, 4, 6]:
        raise ValueError('number of players should be 2, 4 or 6')
    return TockGame(nplayers=nplayers, render=render)


def random_player(obs, info):
    """defines a random player: returns a random action from the actionspace"""
    return choice(info['space'])


def eager_player(obs, info):
    """defines an eager player: returns an action of the most advanced pawn and highest card value"""
    card_to_value = [2, 3, -4, 5, 6, 7, 8, 9, 10, 11, 12, 0, 1]
    maxloc, maxval = -1, -5
    player = info['player']
    for action in info['space']:
        card, pawn = action
        loc = obs[player][pawn]
        val = card_to_value[info['cards'][card]%13]
        if loc >= maxloc and val > maxval:
            maxloc, maxval, bestaction = loc, val, action
          
    return bestaction



def main():
    quit = False
    nplayers = 2
    env = make(nplayers, render=False)
    obs, rew, done, info = env.reset()
    print(env)
    print(info)
    while not done:
        for _ in range(nplayers):
            action = eager_player(obs, info)
            obs, rew, done, info = env.step(action=action)
            # env.render()
            print(env)
            print(info)
            q = input("q to quit")
            if q == 'q':
                quit = True
                break
    print(env)
            
    
    plt.show(block=True)

if __name__ == "__main__":
    main()
