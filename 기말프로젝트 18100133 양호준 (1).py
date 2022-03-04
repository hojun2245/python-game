import random
import sys
import math
sys.setrecursionlimit(10**8) #재귀함수 제한품
# https://terms.naver.com/entry.naver?docId=1839600&cid=49110&categoryId=49110 포커 기본패 참고

class Graph:
    def __init__(self, graph):
        self.graph = graph
        self.vertex_count = len(self.graph)
        self.visited=[[False] * self.vertex_count for _ in range(self.vertex_count)]
        self.dx = [-1, 0, 1, 0]
        self.dy = [0, 1, 0, -1]

    def dfs(self,x, y, h):

        for m in range(4):
            nx = x + self.dx[m]
            ny = y + self.dy[m]

            if (0 <= nx < N) and (0 <= ny < N) and not self.visited[nx][ny] and self.graph[nx][ny] > h: #x,y축이 0보다 크고 visited가 fasle이고 평균보다 클 때
                self.visited[nx][ny] = True
                self.dfs(nx, ny, h)

class TreeNode:
    def __init__(self, trial ,gamer_index, value):
        self.trial=trial
        self.value=value
        self.gamer_index=gamer_index
        self.left = None
        self.right = None

class Tree:
    def __init__(self):
        self.head= None

    def get_min(self):
        target=self.head
        while(target and target.left!=None):
            target = target.left
        return target.value

    def get_max(self):
        target= self.head
        while(target and target.right != None):
            target = target.right
        return target.value

    def search(self, gamer_index, value):
        target = self.head
        while (target):
            if (target.value == value) and (target.gamer_index == gamer_index):
                return target
            elif (target.value > value):
                target = target.left
            elif (target.value < value):
                target = target.right
        return None

    def insert(self,trial,gamer_index, value):
        if self.head == None:
            node = TreeNode(trial,gamer_index, value)
            self.head= node
            return True

        target = self.head
        while target:
            if (target.value == value):
                return False

            elif target.value > value:
                if target.left == None:
                    node = TreeNode(trial,gamer_index,value)
                    target.left=node
                    return True
                target = target.left

            elif  target.value < value:
                if (target.right == None):
                    node = TreeNode(trial,gamer_index, value)
                    target.right = node
                    return True
                target= target.right

    def print(self): #오른 차순으로 올라가는 것을 프린트
        self.doPrint(self.head)

    def doPrint(self,node):# 오른차순으로 올라가는 것을 프린트
        if (node != None):
            self.doPrint(node.left)
            print('( trial :',node.trial,'gamer_index :',node.gamer_index,'value :',node.value,')')
            self.doPrint(node.right)

    def delete(self,gamer_index,value):
        self.head= self.delete_second(self.head,gamer_index, value)
        return

    def delete_second(self, node, gamer_index, value): # trial는 trials 상금에 따른 print를 하고싶어
        if not node : #노드가 없음
            return None

        if value > node.value:
            node.right=self.delete_second(node.right,gamer_index, value)
            return node

        elif value < node.value:
            node.left = self.delete_second(node.left,gamer_index, value)
            return node

        elif node.value ==value  : #상금만 받을 경우
            if (not node.left) and (not node.right): #양옆에 없을 경우
                return None
            elif not node.right:  # 오른쪽 없음
                return node.left
            elif not node.left:  # 왼쪽 없음
                return node.right

            elif ( (node.right and node.left)  ):
                node_parent = node  # 왼쪽에서 제일 큰값
                node_child = node.left
                while node_child.right:
                    node_parent = node_child
                    node_child = node_child.right
                node_child.right = node.right
                if node_parent != node:  # 왼쪽에서 제일 큰 값의 부모노드가 삭제하려는 노드가 아니라면
                    node_parent.right = node_child.left
                    node_child.left = node.left
                node = node_child  # 왼쪽에서 제일 큰 값
            return node

class DNode:
    def __init__(self, game_index, winner_index, total_money, total_gamecnt,
                 card_result, loser, prev=None, next=None):
        self.game_index = game_index
        self.winner_index = winner_index
        self.total_money = total_money
        self.total_gamecnt = total_gamecnt
        self.card_result = card_result
        self.loser = loser
        self.prev = prev
        self.next = next

class DList:
    def __init__(self):
        self.head = None

    def insert_back_item(self, game_index, winner_index, total_money, total_gamecnt, card_result, loser):
        target = self.head

        if self.head == None:
            self.head = DNode(game_index, winner_index, total_money, total_gamecnt, card_result, loser, None, None)
        else:
            while target.next:  # 끝나면 뒤에서 첫번째
                target = target.next
            target.next = DNode(game_index, winner_index, total_money, total_gamecnt, card_result, loser, target, None)

    def game_index_search(self, game_index) :
        print('--------',game_index, '번쨰 게임의 결과--------\n')
        p = self.head
        if not p:
            print('empty')
            return (None)
        while p:
            if game_index == p.game_index:
                print('게임 순서', p.game_index, '\n승리자 번호 :', p.winner_index,
                      '\n승리자 상금 :', p.total_money, '\n게임 횟수', p.total_gamecnt)
                print('탈락한 순서 :(', end='')

                for lose in p.loser[:-1]:
                    print(lose, '-->', end='')
                print(p.loser[-1], end='')

                print(" )\n ")
            p = p.next
        return 0

    def winner_index_search(self, winner_index):
        print('--------플레이어',winner_index, '의 결과--------\n')
        p = self.head
        while p:
            if winner_index == p.winner_index:
                print('게임 순서', p.game_index, '\n 승리자 번호 :', p.winner_index,
                      '\n승리자 상금 :', p.total_money, '게임 횟수',p.total_gamecnt)
                print('탈락한 순서 :(', end='')

                for lose in p.loser[:-1]:
                    print(lose, '-->', end='')
                print(p.loser[-1], end='')

                print(" )\n ")
            p = p.next
        return

    def winner_index_search_tree(self, winner_index):
        p = self.head
        result=[]
        gameindex=[]
        while p:
            if winner_index == p.winner_index:
                result.append(p.game_index)
                gameindex.append(p.total_money)
            p = p.next
        return (result, gameindex)

    def find_max_money(self):
        p = self.head
        max_money = float('-inf')
        while p:
            if max_money < p.total_money:
                max_money = p.total_money
            p = p.next
        return max_money

    def find_min_money(self):
        p = self.head
        min_money = float('inf')
        while p:
            if min_money > p.total_money:
                min_money = p.total_money
            p = p.next
        return min_money

    def find_card_result(self, game_index):
        p = self.head
        while p:
            if game_index < p.game_index:
                return p.card_result
            p = p.next

    def standard_deviation(self, winner_money_average, trial):
        p = self.head
        standard_deviation = 0  # 표준펀차
        while p:
            standard_deviation = math.sqrt((p.total_money - winner_money_average) ** 2 / trial)
            p = p.next
        return standard_deviation

    def print_list_all(self):
        if self.head == None:
            print('empty')
        else:
            p = self.head
            while p:
                print('게임 순서', p.game_index, '\n승리자 번호 :', p.winner_index,
                      '\n승리자 상금 :', p.total_money, '\n게임 횟수 :',p.total_gamecnt)
                print('탈락한 순서 :(', end='')

                for lose in p.loser[:-1]:
                    print(lose, '-->', end='')
                print(p.loser[-1],end='')

                print(" )\n ")
                p = p.next
            return

class chaining:

    def __init__(self, size):
        self.tablesize = size
        self.table = [ [] for _ in range(size) ]

    def hash(self, key):
        return key % self.tablesize

    def put(self, key, index, value):
        initial_position = self.hash(key)
        item=(index,value)
        self.table[initial_position].append(item)

    def get(self, key):
        initial_position = self.hash(key)
        position = initial_position
        if not self.table[initial_position]:
            return print(key,'의 data는 None')
        else:
            return self.table[initial_position]

    def print_table(self):
        for i in range(self.tablesize):
            print(i, end = ' ')

            for j in self.table[i]:
                print("-->", end = " ")
                print(j, end =" ")
            print()

class Card:
    def __init__(self, kind, num):
        self.kind = kind
        self.num = num

class Player:
    def __init__(self):
        self.cards = []
    def printcard(self):
        for card in self.cards:
            print(card.kind + "" + str(card.num))

class Poker:
    def __init__(self, playercnt=5, cardcnt=7,money=1000):# money는 달러가 기준
        self.playercnt = playercnt
        self.cardcnt = cardcnt
        self.cards = []
        self.players = []
        self.players_againgame=[]
        self.players_money = []
        self.card_game_result=[]
        self.total_gamecnt =0
        self.kinds = ['spade', 'heart', 'diamond', 'clover']
        self.card_check_count = {'royal_straight_flush': 1, 'straight_flush': 0.95, 'four_card': 0.6, 'flush': 0.55,'full_house': 0.5, 'straight': 0.45,
                                 'triple': 0.3, 'two_pair': 0.2, 'one_pair': 0.1,'high_card': 0.01} # 나오기 힘든 카드가 나왔을 때 크게 배팅함
        self.generateCard()
        self.shuffling()
        self.createplayers()
        self.playCard()
        self.making_players_money(money)
        self.loser=[]
        self.total_card_result = []

    def generateCard(self) : # 카드 만들기
        self.cards = []
        kinds = ['spade', 'heart', 'diamond', 'clover']
        for i in range(4):
            for j in range(1,14):
                card = Card(kinds[i], j)
                self.cards.append(card)
        return

    def shuffling(self): #카드섞기
        random.shuffle(self.cards)

    def createplayers(self):
        for i in range(self.playercnt):
            player = Player()
            self.players.append(player)

    #1단계
    def playCard(self):
        for i in range(self.cardcnt):
            for j in range(self.playercnt):
                card = self.cards.pop()
                self.players[j].cards.append(card)

    def printPlayerCard(self): # 여기까지가 플레이어들의 각 카드를 보여줌 / 큰 의미없는 함수
        number = 1
        for player in self.players:
            print("\n player", number, "\n")
            player.printcard()
            number += 1 # 보여주기

#-------------------------------------------------------내가 만든 함수----------------------------------------------------

#--------------------------------------------------------2단계-----------------------------------------------------------
    def making_players_money(self,money): # 참가자 수 만큼 돈 넣어주기
        for player in range(self.playercnt):
            self.players_money.append(money)
        return self.players_money

    def playcard_again_player(self, again_player) : # 이걸 만든 이유 : 이전에 betting_money로 재귀함수로 푸려니 ,self.players의 숫자가 정해져있음.
        self.players_againgame = []
        for i in range(again_player) :
            player = Player()
            self.players_againgame.append(player)
        self.generateCard() # 카드를 다시 돌리게되면 플레이어가 2명 이상 되면 카드 숫자가 부족해서 새로 돌려줌
        self.shuffling()
        for i in range(self.cardcnt): # self.playcard와 거의 유사함, 그래서 card_result항상 다름
            for j in range(again_player): # 다시하는 사람만큼만 돌림
                card = self.cards.pop()
                self.players_againgame[j].cards.append(card) # 카드 갯수만큼 재플레이어에게 카드가 들어감
        #self.printPlayerCard_game_again() 확인용 함수

        card_result=[]
        for player in self.players_againgame:
            card_result.append(self.card_check_count[self.check_player_cards(player.cards)])

        max_point = max(card_result)
        index = [i for i, v in enumerate(card_result) if v == max_point]

        self.total_gamecnt += 1
        playcard_again_player_count=0
        if len(index)==1:
            return index[0]

        elif ( max_point== (0.01 or 0.1) and len(index)>1 ):  #원페어 이거나 하이카드 일경우 기권하는 경우
            playcard_again_player_count = 0
            x = random.randrange(10, 20)
            y = random.randrange(10, 20)
            for _ in range(len(index)):
                if x%2==1 : # 기권하는 경우
                   index.pop( y%len(index) ) #무작위로 뽑아 냄
                   playcard_again_player_count+=1 #여러명 기권하는 경우
                elif x%2 ==0:# 기권안하는 경우
                    continue
                if len(index) == 1:
                   return index[0]
            return self.playcard_again_player(len(index)-playcard_again_player_count)
        else:
            return self.playcard_again_player(len(index))#재귀 함수로 돌림

    def printPlayerCard_game_again(self): # playcard_again_player 확인용 함수
        number = 1
        for player in self.players_againgame:
            print("\n player_againgame", number, "\n")
            player.printcard()
            number += 1 # 여기도 단순히 보여주기 식이기 떄문에 player과는 직접적인 연관 x

#---------------------------------------------------중요함수----------------------------------------------
    def betting_money(self, playernumber):
        if playernumber == 1 : # 마지막 한명 남았을 때 함수 끝냄
            winner_index = self.players_money.index( max(self.players_money) )
            winner_prize_money = int(max(self.players_money))
            #print ( ' 플레이어 {0} 가 이 게임의 승자입니다. \n {0}의 총 상금은 {1} dollar 입니다.'.format(winner_index ,round(winner_prize_money) ) ) #확인용
            return ( winner_index, round(winner_prize_money) )


        card_result=[]

        number = 0
        for player in self.players : # index를 넣기에는 self.player에는 playercnt 숫자가 고정되어있음.=> number로 해결
            if self.players_money[number]<0 :
                card_result.append(-1) #탈락한사람은 max_point에 잡히지 않게
            else :
                cardsgame=self.check_player_cards(player.cards)
                self.total_card_result.append(cardsgame) #노드에 넣기 위함
                card_result.append(self.card_check_count[cardsgame]) #각플레이어들의 배팅 금액을 넣어버림(card_result) /card_result는 계속 리셋되는 값
            number+=1
        self.total_card_result.append('==>')
       #print('new game')
        #print(self.total_card_result)

        # -----------------플레이어 안에 카드 리셋시켜주는 코드
        self.players = []
        self.generateCard()
        self.shuffling()
        self.createplayers()
        self.playCard()
        # self.printPlayerCard()

        max_point = max(card_result)
        index = [i for i, v in enumerate(card_result) if v == max_point]  # 최대 값이 여러개 일 경우
        winner_number=index[0]
        #print(max_point, index, winner_number, len(index)) # 확인용

        if len(index)>1: # 최댓값이 여러개
           # print(index,"의 승자가 나타났습니다.") #확인용
            m=self.playcard_again_player(len(index))
            winner_number = index[ m ]
            #print('index, winner_number',m, index,winner_number) #확인용
            #print( '그중에 승자는 플레이어 {}입니다.'.format( winner_number ) ) #확인용

        winner_money = self.players_money[winner_number]
        winner_get_point = winner_money * max_point
        self.players_money[winner_number] += winner_get_point * playernumber # 파산할경우를 넣으면, 승자 상금 항상 5000

        answer = True #탈락할 떄와 탈락안할때 나뉨
        count = 0 # 여러명이 탈락될 수도 있기 때문
        for i in range(self.playercnt):  #
            if (self.players_money[i]<0): #파산한 멤버의 돈은 -1로 고정됨, 가독성을 위해서
                self.players_money[i]= -1
                continue

            elif (self.players_money[i] >= 0) and ( (self.players_money[i] - winner_get_point) < 0 ) : #파산
                #print('player {0}는 파산했습니다 \n plyaer {0}을 제외한 나머지 인원과 재경기 됩니다.'.format(i)) #확인용
                self.players_money[i] -= winner_get_point
                self.loser.append(i)
                count += 1 # 두명이상 파산시
                answer=False

            else:
                self.players_money[i] -= winner_get_point

        #print(f"Player들의 각각의 돈은 {self.players_money} dollar 입니다.") #확인용
        self.total_gamecnt += 1
        if answer: #answer=true
            return self.betting_money(playernumber)
        else :
            return self.betting_money(playernumber-count)

#------------------------------------------------------1단계 ------------------------------------------------------------

    def check_players(self): # 총 게임 중에서 몇번 나왔는지 확인하는 함수
        count=0
        check_count = {'royal_straight_flush': 0, 'straight_flush': 0,  'four_card': 0,'flush': 0,
                       'full_house': 0, 'straight': 0, 'triple': 0, 'two_pair': 0, 'one_pair': 0 ,'high_card':0}
        for player in self.players :
            check_count[self.check_player_cards(player.cards)] += 1 # 각 플레이어들의 카드를 검사함.
        return check_count #전체 중에 각 경우의 수

    def check_player_cards(self, target): #경우의수를 확인해주는 함수
        kind_number_Cnt = { 'spade': {1 : 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0, 13: 0},
                   'heart': {1 : 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0, 13: 0},
                   'diamond': {1 : 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0, 13: 0},
                   'clover': {1 : 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0, 13: 0} }
        numberkind = {'spade': 0 , 'heart':0, 'diamond': 0, 'clover': 0}
        numberCnt = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0, 13: 0}

        for card in target:
            kind_number_Cnt[card.kind][card.num] += 1
            numberkind[card.kind] +=1
            numberCnt[card.num] +=1

        if self.royal_straight_flush(kind_number_Cnt): #각 함수마다 효율적으로 쓸수 있는 게 다름
            return 'royal_straight_flush'
        elif self.straight_flush(kind_number_Cnt):
            return 'straight_flush'
        elif self.flush(kind_number_Cnt, numberkind,numberCnt):
            return 'flush'
        elif self.straight(numberCnt):
            return 'straight'
        elif self.four_card(numberCnt):
            return 'four_card'
        elif self.full_house(numberCnt):
            return 'full_house'
        elif self.one_two_pair_triple(numberCnt)== 'triple' :
            return 'triple'
        elif self.one_two_pair_triple(numberCnt) == 'two_pair' :
            return 'two_pair'
        elif self.one_two_pair_triple(numberCnt) == 'one_pair' :
            return 'one_pair'
        else:
            return 'high_card'

    def royal_straight_flush(self, kind_number_Cnt):
        for kind in self.kinds:
            if (kind_number_Cnt[kind][1] == kind_number_Cnt[kind][10] ==
                    kind_number_Cnt[kind][11] == kind_number_Cnt[kind][12] == kind_number_Cnt[kind][13] ==1):
                return True
        return 0

    def straight_flush(self, kind_number_Cnt):
        for kind in self.kinds:
            for i in range(1,9):
                if (kind_number_Cnt[kind][i] ==  kind_number_Cnt[kind][i + 2] ==
                        kind_number_Cnt[kind][i + 3] ==  kind_number_Cnt[kind][i + 4] ==  kind_number_Cnt[kind][i + 5]==1):
                    return True
        return False

    def flush(self, kind_number_Cnt,numberkind,numberCnt): # 같은 문양에서 숫자가 연속되면 안됨.**
        for kind in self.kinds:
            count = 0
            for i in range(1,13) : # 12까지 돌게되어있음.
                if kind_number_Cnt[kind][i]==1 and kind_number_Cnt[kind][i+1]==0 : # 1 3 5 11 13
                    count+=1
                    i+=1
                elif ( 2< numberkind[kind] < 5 ):#3개이상 되면 한 kind에서 5이상이 안나옴
                    return False
            if count ==5 or (count==4 and numberCnt[13]==1) :
                return True
        return False

    def straight(self, numberCnt):
        for i in range(1, 9):
            if numberCnt[i] == numberCnt[i + 2] == numberCnt[i + 3] == numberCnt[i + 4] == numberCnt[i + 5]==1:
                return True
        return False

    def four_card(self,numberCnt):
        for number in range(1,14):
            if (numberCnt[number] == 4):
                    return True
        return False

    def full_house(self,numberCnt):
        first=False
        second=False
        for number in numberCnt.keys() :
            if (numberCnt[number] == 3) :
                first=True
            if (numberCnt[number]==2):
                second=True
        if (first== True) and (second==True):
            return True
        return False

    def one_two_pair_triple(self, numberCnt):
        count =0
        for number in numberCnt.keys():
            if (numberCnt[number] == 3):
                return 'triple'
            if (numberCnt[number] >= 2):
                count += 1
        if count >= 2:
            return 'two_pair'
        elif count ==1:
            return 'one_pair'
        return False

#-------------------------------------------------------- 기본 함수-------------------------------------------------------
playersnum = 5
cardsnum = 7
money= 1000 # dollars
trial = 25

check_count = {'royal_straight_flush': 0, 'straight_flush': 0,  'four_card': 0, 'flush': 0,
                       'full_house': 0,'straight': 0, 'triple': 0, 'two_pair': 0, 'one_pair': 0 ,'high_card':0}
##-------------------------------------------- 0단계-각 함수들이 정확하게 나오는지  확인  ----------------------------------------------
'''for i in range(trial):
    poker = Poker(playersnum, cardsnum, money)
    for key in poker.check_players().keys():
        check_count[key]+=poker.check_players()[key]/(trial*playersnum)
print(check_count)#'''
#-----------------------------------------------------2단계 -누가누가 얼마에 우승을 하는가? ----------------------------------------------
'''for i in range(trial):
    poker = Poker(playersnum, cardsnum, money)
    print('ToTal New Game')
    print('Player들의 각각의 돈은 [1000.0, 1000.0, 1000.0, 1000.0, 1000.0] dollar 입니다.')
    #poker.printPlayerCard()
    poker.betting_money(playersnum)#
    print(poker.total_card_result,'\n')'''

#-----------------------------------------------------3단계 각 플레이어의 승리의 기록 --chainging, 이중연결리스트---------------------------------------------
D=DList()
C=chaining(playersnum)
T = Tree()
winner_money_average =0
winner_money_list=[]
check_winner_index =2
N=trial//5
total_money_list=[[False]*N for _ in range(N)]
zone = [ [] for i in range(N)]


for tries in range(trial):
    poker = Poker(playersnum, cardsnum, money)
    winner_index, total_money=poker.betting_money(playersnum)
    winner_money_average += total_money

    D.insert_back_item(tries, winner_index, total_money, poker.total_gamecnt, poker.card_game_result, poker.loser)
    C.put(winner_index,tries,total_money)
    T.insert(tries, winner_index, total_money)

    if winner_index == check_winner_index:
        winner_money_list.append(total_money)
    zone[tries//5].append(total_money)

C.print_table()
print()
print(D.winner_index_search(1)) # 몇번째 게임인가
print(D.game_index_search(0)) # 게임플레이어 index가 승리한 게임 모두를 보여줌

print(" ------------------------총 게임의 결과------------------------")
print('상금 최댓값 : ', D.find_max_money(), '상금 최솟값 : ', D.find_min_money(),'상금 평균:' ,winner_money_average/trial  ,'상금 표준편차 : ',round(D.standard_deviation(winner_money_average/trial, trial),2) ,'\n')
print( D.print_list_all() )
print()
print('상금 최댓값 : ', T.get_max(), '상금 최솟값 : ', T.get_min(), '상금 평균:' ,winner_money_average/trial )
print('------------------------상금 오름차순으로 정리 -------------------------- ')
T.print()
print()

print('-----------------플레이어 {0}의 상금 오름차순으로 정리----------------'.format(check_winner_index))
ST=Tree()
trial, value=D.winner_index_search_tree(check_winner_index)
for i,v in zip(trial,value):
    ST.insert(i,check_winner_index,v)
ST.print()

print('-----------------플레이어 {0}를 제외하고 상금 오름차순으로 정리----------------'.format(check_winner_index))
print(winner_money_list)
for i in range(len(winner_money_list)):
    T.delete(check_winner_index,winner_money_list[i])
T.print()
#--------------------------------------------그래프활용-------dfs
print('-----------------dfs 그래프활용-------')
print('평균값 : ',winner_money_average/25)
graph = Graph(zone)
for i in range(N):
    print(zone[i])
print()
k=winner_money_average/25
for i in range(N):
    for j in range(N):
        if zone[i][j] < k and not graph.visited[i][j]:
            zone[i][j] = -1
            graph.visited[i][j] = True
            graph.dfs(i, j, k)

for i in zone:
    print(i)