## Neighbours after a recall
After recall_memory, the neighbours of the items it showed are offered after them, one relation hop at a time, and after each you answer one line, "[h1.2] | useful (yes or no):": yes when the neighbour holds something the question needs that the items above do not already give (often the document a fact came from, or a fact one relation away); no for anything else. Only the useful neighbours stay, and only they are followed to the next hop.
Examples (made up, not in memory):
Question: Where did Kel Aster train before opening the Rowan bakery?
[h1.1] [datasource] derived_from from 3f2a1b0c
  id: 11111111-0000-0000-0000-00000000000a
  content: Tarn Gazette, June 2018: Kel Aster, who trained for four years at the Pell bakery school, opens the Rowan bakery on Tarn Street.
[h1.1] | useful (yes or no): yes
[h1.2] [fact] similar_to to 3f2a1b0c
  id: 11111111-0000-0000-0000-00000000000b
  content: The Tarn Street market runs every Saturday from 8 to 1.
[h1.2] | useful (yes or no): no
