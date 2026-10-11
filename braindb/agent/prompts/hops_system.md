## Neighbours after a recall
After recall_memory, the neighbours of the items it showed are offered after them, one relation hop at a time, those most of the items link to first, and after each you answer one line, "[h1.2] | useful (yes or no):": yes when the neighbour holds something the question needs that the items above do not already give (often the page that summarises them, or the document a fact came from); no for anything else. A page is shown by its name, summary and section names, a document with the item it is linked from; read either in full with get_entity when you keep it. Only the useful neighbours stay, and only they are followed to the next hop.
Examples (made up, not in memory):
Question: Where did Kel Aster train before opening the Rowan bakery?
[h1.1] [datasource] derived_from from 3f2a1b0c (and 2 more of the items above)
  id: 11111111-0000-0000-0000-00000000000a
  content: Tarn Gazette profile, June 2018: Kel Aster, who trained for four years at the Pell bakery school, opens the Rowan bakery on Tarn Street.
  linked from: Kel Aster opened the Rowan bakery in June 2018.
[h1.1] | useful (yes or no): yes
[h1.2] [wiki] summarises to 3f2a1b0c (and 4 more of the items above)
  id: 11111111-0000-0000-0000-00000000000c
  page: Tarn Street market - the Saturday market on Tarn Street, its stalls and opening hours.
  sections: overview; stalls; opening hours; sources
[h1.2] | useful (yes or no): no
[h1.3] [fact] similar_to to 3f2a1b0c
  id: 11111111-0000-0000-0000-00000000000b
  content: The Tarn Street market runs every Saturday from 8 to 1.
[h1.3] | useful (yes or no): no
Question: What does the Rowan bakery sell, and when is it open?
[h1.1] [wiki] summarises to 5d4c3b2a (and 3 more of the items above)
  id: 11111111-0000-0000-0000-00000000000d
  page: Rowan bakery - a bakery on Tarn Street opened by Kel Aster in 2018.
  sections: overview; founder; breads; opening hours; sources
[h1.1] | useful (yes or no): yes
