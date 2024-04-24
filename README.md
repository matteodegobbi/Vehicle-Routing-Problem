# Vehicle-Routing-Problem

## Idee
- Assumiamo il grafo sia completo. Questp e' senza perdita di generalita' in quanto se avessimo in input un grafo non completo basterebbe eseguire un shortest path (tipo Djikstra) da ogni nodo e aggiungere una edge tra tutti i nodi con costo uguale a quello della shortest path (a patto che stiamo partendo da un grafo connesso, condizione necessaria per VRP).
- Ogni cliente ha un numero intero che lo rappresenta univocamente, possiamo assegnare questo numero casualmente oppure con qualche euristica intelligente.
- Anche i veicoli sono rappresentati da un numero intero.
- Le varie soluzioni le rappresentiamo come un vettore di interi, il cui indice rappresenta il numero del cliente e il valore contenuto nella cella e' il numero del veicolo assegnato a quel cliente.
- L'assegnazione iniziale dei clienti ai veicoli puo' essere random oppure in maniera intelligente possiamo usare qualcosa del tipo kmeans++ per partire da una soluzione dove i veicoli servono clienti vicini tra loro.
- Per il GA possiamo usare una funzione di fitness basata sulla distanza totale percorsa (carburante consumato), sul vincolo di avere cicli Hamiltoniani e sulla capacita' massima dei veicoli.
- Per il GA usiamo ricombinazione e mutazioni, potremmo anche includere elitismo per migliorare le performance, per possibili tipi di ricombinazione guardiamo sul paper.
- Ripetere il GA con varie random initialization e scegliere la soluzione migliore.
- Confronto con Simulated Annealing oppure con Tabu search o altri sota
- Per rappresentare il grafo usiamo NetworkX su python.
