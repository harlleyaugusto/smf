#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <climits>

#include <string>
#include <algorithm>
#include <vector>
#include <array>
#include <map>
#include <set>
#include <random>
#include <cmath>
#include <random>
#include <numeric>
#include <stdlib.h>
#include <iomanip>

#include <ctime>
#include <bitset>

#define DELIM ','
#define EPS 1e-8

#define BETA_alpha 1.0
#define BETA_beta 1.0

#define MATRIX_FACTORIZATION 13

#define NONE_SIMILARITY 0

#define PEARSON_SIMILARITY 1
#define JACCARD_SIMILARITY 2
#define JACCARD_ASYMMETRIC_SIMILARITY 21
#define COSINE_SIMILARITY 3
#define ADAMIC_ADAR_SIMILARIY 4

#define INTERSECTION 100

#define sigmoid(x) (1/(1+exp(-x)))

#define ALPHA 1e-4

#define RELEVANCE_LIKED 2
#define RELEVANCE_N_LIKED 1
#define TRAINING_LIKED 0
#define TRAINING_N_LIKED 0
#define IRRELEVANT 0

using namespace std;

// #################### BEGIN DATA STRUCTURES

unsigned int similarity_type;
unsigned int K_FOLD;
unsigned int BEGIN_K_FOLD;
unsigned int END_K_FOLD;
vector<unsigned int> k_count;
vector<unsigned int> topN;

typedef struct {
	unsigned int itemId;
	unsigned int userId;
	float rating;
	short iRating;
} Vote;

struct Item {
	Item() :
			// dummy votes
			nUps(1), nRatings(2) {
	}
	~Item() {
		users.clear();
		ratings.clear();
	}

	unsigned int id;
	unsigned int nUps;
	unsigned int nRatings;
	double posteriori_success;
	double odds_ratio;
	double avg;
	set<unsigned int> users;
	vector<Vote *> ratings;
	double adamic_adar;
	double adamic_adar_sum_neighbors;
};
typedef struct Item Item;

struct User {
	unsigned int id;
	set<unsigned int> items;
	vector<Vote *> ratings;
	double avg;
	double std;
	double adamic_adar;
	double adamic_adar_sum_neighbors;

	~User() {
		for (size_t i = 0; i < ratings.size(); ++i)
			delete ratings[i];
		ratings.clear();
		items.clear();
	}
};
typedef struct User User;

typedef struct {
	unsigned int userId;
	unsigned int itemId;
	float rating;
	unsigned short int fold;
} Review;

unsigned int NUM_THREADS = 1;

float MAX_RATING;
float MIN_RATING;
float DELTA_RATING;
vector<Review*> reviews;
vector<User*> users;
vector<Item*> items;

map<unsigned int, unsigned int> userIds; // Maps real user's ID to id
map<unsigned int, unsigned int> itemIds; // Maps real item's ID to id

map<unsigned int, unsigned int> rUserIds; // Inverse of the above maps
map<unsigned int, unsigned int> rItemIds;

std::mt19937 generator;

struct Gen {
	Gen() :
			n(K_FOLD) {
		v.resize(K_FOLD, 0);
		fill();
	}
	vector<unsigned int> v;
	unsigned int n;

	void fill() {
		n = K_FOLD;
		for (unsigned int i = 0; i < K_FOLD; ++i)
			v[i] = i;
	}

	unsigned int next() {
		if (n < 1)
			fill();
		int i = rand() % n;
		unsigned int number = v[i];
		unsigned int temp = v[n - 1];
		v[n - 1] = number; //TODO: this line is not necessary
		v[i] = temp;
		n--;
		return number;
	}

};
typedef struct Gen Gen;

typedef struct {
	User *user;
	vector<Review *> reviews;
} TestItem;

// ### END DATA STRUCTURE

//### BEGIN MISCELLANEOUS
int cmp(double x, double y, double tol = 1e-19) {
	return (x <= y + tol) ? (x + tol < y) ? -1 : 0 : 1;
}
//### END MISCELLANEOUS

//#### BEGIN METRICS

double hitsAt(vector<pair<double, unsigned int> > &ranked_items,
		set<unsigned int> &correct_items, set<unsigned int> &ignore_items,
		unsigned int n) {

	int hit_count = 0;
	int left_out = 0;

	for (unsigned int i = 0; i < ranked_items.size() && i < n + left_out; i++) {
		unsigned int item_id = ranked_items[i].second;
		if (ignore_items.find(item_id) != ignore_items.end()) {
			++left_out;
			continue;
		}

		if (correct_items.find(item_id) != correct_items.end()) {
			hit_count++;
		}
	}

	return hit_count;
}

vector<vector<double> > getPrecisionAndRecall(
		vector<pair<double, unsigned int> > &ranked_items,
		set<unsigned int> &correct_items, set<unsigned int> &ignore_items) {

	vector<vector<double> > result(2);

	for (unsigned int i = 0; i < topN.size(); ++i) {
		unsigned int k = topN[i];

		double hits = ((double) hitsAt(ranked_items, correct_items,
				ignore_items, k));
		double precision = hits / k;
		double recall = hits / correct_items.size();

		result[0].push_back(precision);
		result[1].push_back(recall);

	}
	return result;
}

double getMeanReciprocalRank(vector<pair<double, unsigned int> > &ranked_items,
		set<unsigned int> &correct_items, set<unsigned int> &ignore_items) {

	for (unsigned int i = 0; i < ranked_items.size(); i++) {
		unsigned int item_id = ranked_items[i].second;
		if (ignore_items.find(item_id) != ignore_items.end()) {
			continue;
		}
		if (correct_items.find(item_id) != correct_items.end()) {
			return 1.0 / (i + 1);
		}
	}
	return 0;
}

double computeIDCG(unsigned int n) {
	double idcg = RELEVANCE_LIKED;
	for (unsigned int i = 1; i < n; i++) {
		idcg += RELEVANCE_LIKED / log2(i + 1);
	}
	return idcg;
}

double computeNDCG(vector<pair<double, unsigned int> > &ranked_items,
		set<unsigned int> &liked_items, set<unsigned int> &notliked_items,
		set<unsigned int> &ignore_items, unsigned int n) {
	double dcg = 0;
	double idcg = computeIDCG(n);
	int left_out = 0;

	for (unsigned int i = 0; i < n + left_out; i++) {
		unsigned int item_id = ranked_items[i].second;
		if (ignore_items.find(item_id) != ignore_items.end()) {
			left_out++;
			continue;
		}

		unsigned int relevance = IRRELEVANT;

		if (liked_items.find(item_id) != liked_items.end()) {
			relevance = RELEVANCE_LIKED;
		} else if (notliked_items.find(item_id) != notliked_items.end()) {
			relevance = RELEVANCE_N_LIKED;
		}
		// compute NDCG part
		int rank = i + 1 - left_out;
		if (rank != 1) {
			dcg += relevance / log2(rank + 1);
		} else {
			dcg += relevance;
		}
	}

	return dcg / idcg;
}

double AP(vector<pair<double, unsigned int> > &ranked_items,
		set<unsigned int> &correct_items, set<unsigned int> &ignore_items) {

	unsigned int hit_count = 0;
	double avg_prec_sum = 0;
	unsigned int left_out = 0;

	for (unsigned int i = 0; i < ranked_items.size(); i++) {
		unsigned int item_id = ranked_items[i].second;
		if (ignore_items.find(item_id) != ignore_items.end()) {
			left_out++;
			continue;
		}

		if (correct_items.find(item_id) != correct_items.end()) {
			hit_count++;

			avg_prec_sum += ((double) hit_count) / (i + 1 - left_out);
		}

	}

	if (hit_count != 0) {
		return avg_prec_sum / hit_count;
	} else {
		return 0;
	}
}
//#### END METRICS

// #################### BEGIN EXPERIMENT

struct Experiment {
	Experiment() {
		precision.resize(topN.size(), 0);
		recall.resize(topN.size(), 0);
		ndcg.resize(topN.size(), 0);
		reciprocal = 0;
		map = 0;
	}

	double map;
	double reciprocal;

	vector<double> precision;
	vector<double> recall;
	vector<double> ndcg;

	void print() {
		cout << setprecision(6) << map << "," << reciprocal << ",";

		size_t n = topN.size();
		for (size_t i = 0; i < n; ++i) {
			cout << setprecision(6) << precision[i] << ",";
		}

		for (size_t i = 0; i < n; ++i) {
			cout << setprecision(6) << recall[i] << ",";
		}

		for (size_t i = 0; i < n; ++i) {
			cout << setprecision(6) << ndcg[i] << ",";
		}
		cout << endl;
	}

	void normalize(unsigned int test_size) {
		for (size_t j = 0; j < topN.size(); ++j) {
			precision[j] /= test_size;
			recall[j] /= test_size;
			ndcg[j] /= test_size;
		}

		map /= test_size;
		reciprocal /= test_size;
	}

};
typedef struct Experiment Experiment;

Experiment mergeExperiment(const vector<Experiment> &vect) {
	Experiment result;

	for (size_t i = 0; i < vect.size(); ++i) {
		result.map += vect[i].map;
		result.reciprocal += vect[i].reciprocal;

		for (size_t j = 0; j < topN.size(); ++j) {
			result.ndcg[j] += vect[i].ndcg[j];
			result.precision[j] += vect[i].precision[j];
			result.recall[j] += vect[i].recall[j];
		}
	}
	return result;
}

string doExperimentAll(User *u, vector<Review *> &reviews, vector<double> &rank,
		set<unsigned int> &liked, set<unsigned int> &not_liked,
		Experiment &result) {

	// construindo novo rank
	vector<pair<double, unsigned int> > uRank;
	for (size_t i = 0; i < items.size(); ++i) {
		uRank.push_back(make_pair(rank[i], i));
	}
	sort(uRank.rbegin(), uRank.rend());

	ostringstream ostr;
	ostr << u->id << "," << u->ratings.size() << "," << reviews.size() << ",";

	vector<vector<double> > pr = getPrecisionAndRecall(uRank, liked, u->items);
	for (size_t i = 0; i < topN.size(); ++i) {
		result.precision[i] += pr[0][i];

		ostr << setprecision(5) << pr[0][i] << ",";
	}

	for (size_t i = 0; i < topN.size(); ++i) {
		result.recall[i] += pr[1][i];

		ostr << setprecision(5) << pr[1][i] << ",";
	}

	double aux = AP(uRank, liked, u->items);
	result.map += aux;
	ostr << setprecision(5) << aux << ",";

	for (size_t i = 0; i < topN.size(); ++i) {
		unsigned int k = topN[i];
		aux = computeNDCG(uRank, liked, not_liked, u->items, k);

		result.ndcg[i] += aux;
		ostr << setprecision(5) << aux << ",";
	}

	aux = getMeanReciprocalRank(uRank, liked, u->items);
	result.reciprocal += aux;
	ostr << setprecision(5) << aux << ",";

	return ostr.str();
}

void print_result(Experiment &exp, string s) {
	ofstream outfile;
	outfile.open(s.c_str(), std::ios_base::app);

	outfile << setprecision(6) << exp.map << "," << exp.reciprocal << ",";

	size_t n = topN.size();
	for (size_t i = 0; i < n; ++i) {
		outfile << setprecision(6) << exp.precision[i] << ",";
	}

	for (size_t i = 0; i < n; ++i) {
		outfile << setprecision(6) << exp.recall[i] << ",";
	}

	for (size_t i = 0; i < n; ++i) {
		outfile << setprecision(6) << exp.ndcg[i] << ",";
	}
	outfile << endl;
	outfile.close();
}

// ################## END EXPERIMENT

// ################# BEGIN SIMILARITY
bool voteComparator(const Vote * l, const Vote * r) {
	return l->itemId < r->itemId;
}

bool voteComparator2(const Vote * l, const Vote * r) {
	return l->userId < r->userId;
}

double vct_norm(vector<float>& v) {
	double sum_val = 0;
	size_t size = v.size();
	for (size_t i = 0; i < size; ++i)
		sum_val += pow(v[i], 2);
	return sqrt(sum_val);
}

vector<double> similarity_matrix;

// item similarity

double calc_similarity_item(unsigned int p, unsigned int q, unsigned int type) {

	vector<Vote *> u = items[p]->ratings;
	vector<Vote *> v = items[q]->ratings;

	vector<float> p_v;
	vector<float> q_v;

	vector<unsigned> intersection;

	unsigned int i = 0, j = 0, n = 0;
	while (i < u.size() && j < v.size()) {
		if (u[i]->userId < v[j]->userId)
			i++;
		else if (v[j]->userId < u[i]->userId)
			j++;
		else {
			p_v.push_back(u[i]->rating);
			q_v.push_back(v[j]->rating);
			intersection.push_back(u[i]->userId);

			++n;
			++i;
			++j;
		}

	}

	if (p_v.size() == 0) {
		return 0.0;
	}

	double value;
	switch (type) {
	case PEARSON_SIMILARITY: {
		// pearson
		double num = 0;
		double x_den = 0;
		double y_den = 0;
		for (unsigned int i = 0; i < p_v.size(); ++i) {
			double a = (p_v[i] - items[p]->avg);
			double b = (q_v[i] - items[q]->avg);
			num += a * b;
			x_den += pow(a, 2);
			y_den += pow(b, 2);
		}
		value = (num / sqrt(x_den * y_den));
		break;
	}
	case JACCARD_SIMILARITY: {
		// number of intersection
		value = ((double) p_v.size()) / (u.size() + v.size() - p_v.size());
		break;
	}
	case JACCARD_ASYMMETRIC_SIMILARITY: {
		value = ((double) p_v.size()) / (u.size());
		break;
	}
	case COSINE_SIMILARITY: {
		// cosine
		double num = 0;
		double x_den = vct_norm(p_v);
		double y_den = vct_norm(q_v);
		for (unsigned int i = 0; i < p_v.size(); ++i) {
			num += p_v[i] * q_v[i];
		}
		value = (num / sqrt(x_den * y_den));

		break;
	}
	case ADAMIC_ADAR_SIMILARIY: {
		value = 0;
		for (unsigned i = 0; i < intersection.size(); ++i) {
			value += users[intersection[i]]->adamic_adar;
		}

		value = value / users[intersection[i]]->adamic_adar_sum_neighbors;

		break;
	}
	case INTERSECTION:
		value = p_v.size();
		break;
	}

	return value;
}

double get_similarity_item(unsigned int p, unsigned int q, unsigned int type) {

	if (p == q) {
		return 1;
	}

	if (p > q) {
		int aux = p;
		p = q;
		q = aux;
	}
	unsigned int index = items.size() * p + q - 0.5 * ((p + 2) * (p + 1));

	if (similarity_matrix[index] < 0) {
		similarity_matrix[index] = calc_similarity_item(p, q, type);
	}

	return similarity_matrix[index];

}

// user

double calc_similarity_user(unsigned int p, unsigned int q, unsigned int type) {

	User *u = users[p];
	User *v = users[q];

	vector<float> x_v;
	vector<float> y_v;

	vector<unsigned> intersection;

	unsigned int i = 0, j = 0;
	while (i < u->ratings.size() && j < v->ratings.size()) {
		if (u->ratings[i]->itemId < v->ratings[j]->itemId)
			i++;
		else if (v->ratings[j]->itemId < u->ratings[i]->itemId)
			j++;
		else {
			x_v.push_back(u->ratings[i]->rating);
			y_v.push_back(v->ratings[j]->rating);

			intersection.push_back(u->ratings[i]->itemId);
			++i;
			++j;
		}
	}

	if (x_v.size() == 0) {
		return 0.0;
	}

	double value;

	switch (type) {
	case PEARSON_SIMILARITY: {
		// pearson
		double num = 0;
		double x_den = 0;
		double y_den = 0;
		for (unsigned int i = 0; i < x_v.size(); ++i) {
			double a = (x_v[i] - u->avg);
			double b = (y_v[i] - v->avg);
			num += a * b;
			x_den += pow(a, 2);
			y_den += pow(b, 2);
		}
		value = (num / sqrt(x_den * y_den));
		break;
	}
	case JACCARD_SIMILARITY: {
		// number of intersection
		value = ((double) x_v.size())
				/ (u->ratings.size() + v->ratings.size() - x_v.size());
		break;
	}
	case JACCARD_ASYMMETRIC_SIMILARITY: {
		value = ((double) x_v.size()) / (u->ratings.size());
		break;
	}
	case COSINE_SIMILARITY: {
		// cosine
		double num = 0;
		double x_den = vct_norm(x_v);
		double y_den = vct_norm(y_v);
		for (unsigned int i = 0; i < x_v.size(); ++i) {
			num += x_v[i] * y_v[i];
		}
		value = (num / sqrt(x_den * y_den));

		break;
	}

	case ADAMIC_ADAR_SIMILARIY: {
		value = 0;
		for (unsigned i = 0; i < intersection.size(); ++i) {
			value += items[intersection[i]]->adamic_adar;
		}

		value = value / items[intersection[i]]->adamic_adar_sum_neighbors;

		break;
	}

	case INTERSECTION:
		value = x_v.size();
		break;
	}

	return value;
}

double get_similarity_user(unsigned int p, unsigned int q, unsigned int type) {

	if (type == 0) {
		return 1;
	}

	if (p == q) {
		return 1;
	}

	if (p > q) {
		int aux = p;
		p = q;
		q = aux;
	}
	unsigned int index = users.size() * p + q - 0.5 * ((p + 2) * (p + 1));

	if (similarity_matrix[index] < 0) {
		similarity_matrix[index] = calc_similarity_user(p, q, type);
	}

	return similarity_matrix[index];
}

// ################# END SIMILARITY

// ######### MATRIX FACTORIZATION

unsigned int MF_ALGORITHM = 0;
unsigned int MF_NUM_FACTORS = 10;
unsigned int MF_NUM_ITERATIONS = 10;
double MF_LAMBDA = 100;
double MF_ALPHA = 0.01;
unsigned int MF_NORMALIZE = 0;
unsigned int MF_SIMILARITY_USER = 0;
unsigned int MF_SIMILARITY_ITEM = 0;

double squared_norm(const vector<double> &v) {
	double sum = 0;
	for (size_t i = 0; i < v.size(); ++i) {
		sum += pow(v[i], 2);
	}
	return sum;
}

double dot_product(const vector<double> &p, const vector<double> &q) {
	double s = 0.0;
	for (unsigned int k = 0; k < p.size(); ++k) {
		s += p[k] * q[k];
	}
	return s;
}

typedef struct {
	unsigned int u;
	unsigned int v;
	double s;
	double weight;
} pair_similarity;

void sgd_smf(const vector<Vote *> &trainingset, vector<vector<double> > &p,
		vector<vector<double> > &q) {

	cout << "BEGIN MF" << endl;

	const unsigned int SIZE = trainingset.size();

	vector<unsigned int> training_indexes(SIZE, 0);
	for (unsigned int i = 0; i < SIZE; ++i) {
		training_indexes[i] = i;
	}

	double threshold = 0.0;
	vector<pair_similarity> neighborhood_user;
	vector<unsigned int> pair_indexes_user;

	vector<pair_similarity> neighborhood_item;
	vector<unsigned int> pair_indexes_item;

	if (MF_SIMILARITY_USER) {

		int count = 0;
		for (unsigned int u = 0; u < users.size(); ++u) {
			for (unsigned v = u + 1; v < users.size(); ++v) {
				double value = calc_similarity_user(u, v, MF_SIMILARITY_USER);
				if (abs(value) > threshold) {
					pair_similarity pair;
					pair.u = u;
					pair.v = v;
					pair.s = value;
					//pair.weight = calc_similarity_user(u, v, INTERSECTION) + 1;

					pair.weight = 1;

					neighborhood_user.push_back(pair);
					pair_indexes_user.push_back(count++);
				}
			}
		}
	}

	if (MF_SIMILARITY_ITEM) {
		//cout << "ITEM" << endl;
		int count = 0;
		for (unsigned int u = 0; u < items.size(); ++u) {
			for (unsigned v = u + 1; v < items.size(); ++v) {
				double value = calc_similarity_item(u, v, MF_SIMILARITY_ITEM);
				if (abs(value) > threshold) {
					pair_similarity pair;
					pair.u = u;
					pair.v = v;
					pair.s = value;
					//pair.weight = calc_similarity_item(u, v, INTERSECTION) + 1;
					pair.weight = 1;

					neighborhood_item.push_back(pair);
					pair_indexes_item.push_back(count++);
				}
			}
		}
		//cin.get();
	}

	for (unsigned int it = 0; it < MF_NUM_ITERATIONS; ++it) {
		cout << it << endl;
		random_shuffle(training_indexes.begin(), training_indexes.end());

		for (unsigned int i = 0; i < SIZE; ++i) {
			Vote *v = trainingset[training_indexes[i]];

			unsigned int u_id = v->userId;
			unsigned int i_id = v->itemId;

			double r = v->rating;
			double err = (r - dot_product(p[u_id], q[i_id]));

			if (MF_NORMALIZE) {
				r = (v->rating - MIN_RATING) / DELTA_RATING;
				double sig_p = sigmoid(dot_product(p[u_id], q[i_id]));
				err = (r - sig_p) * sig_p * (1 - sig_p);
			}

			for (unsigned int k = 0; k < MF_NUM_FACTORS; ++k) {
				p[u_id][k] += MF_ALPHA
						* (err * q[i_id][k] - MF_LAMBDA * p[u_id][k]);
				q[i_id][k] += MF_ALPHA
						* (err * p[u_id][k] - MF_LAMBDA * q[i_id][k]);
			}
		}

		if (MF_SIMILARITY_USER) {

			random_shuffle(pair_indexes_user.begin(), pair_indexes_user.end());
			for (unsigned int i = 0; i < pair_indexes_user.size(); ++i) {

				unsigned int idx = pair_indexes_user[i];

				const unsigned int u = neighborhood_user[idx].u;
				const unsigned int v = neighborhood_user[idx].v;
				const double s = neighborhood_user[idx].s;
				const double w = neighborhood_user[idx].weight;

				double sig_p = dot_product(p[u], p[v]);
				double err = (s - sig_p);
				if (MF_NORMALIZE) {
					sig_p = sigmoid(dot_product(p[u], p[v]));
					err = (s - sig_p) * sig_p * (1 - sig_p);
				}

				for (unsigned int k = 0; k < MF_NUM_FACTORS; ++k) {
					p[u][k] += MF_ALPHA * (w * err * p[v][k]);
					p[v][k] += MF_ALPHA * (w * err * p[u][k]);
				}
			}
		}

		if (MF_SIMILARITY_ITEM) {

			random_shuffle(pair_indexes_item.begin(), pair_indexes_item.end());
			for (unsigned int i = 0; i < pair_indexes_item.size(); ++i) {

				unsigned int idx = pair_indexes_item[i];
				const unsigned int u = neighborhood_item[idx].u;
				const unsigned int v = neighborhood_item[idx].v;
				const double s = neighborhood_item[idx].s;
				const double w = neighborhood_item[idx].weight;

				double sig_p = dot_product(q[u], q[v]);
				double err = (s - sig_p);
				if (MF_NORMALIZE) {
					sig_p = sigmoid(dot_product(q[u], q[v]));
					err = (s - sig_p) * sig_p * (1 - sig_p);
				}

				for (unsigned int k = 0; k < MF_NUM_FACTORS; ++k) {
					q[u][k] += MF_ALPHA * (w * err * q[v][k]);
					q[v][k] += MF_ALPHA * (w * err * q[u][k]);
				}
			}
		}
	}
}

void sgd_smf_asymmetric(const vector<Vote *> &trainingset,
		vector<vector<double> > &p, vector<vector<double> > &q) {

	cout << "BEGIN MF" << endl;

	const unsigned int SIZE = trainingset.size();

	vector<unsigned int> training_indexes(SIZE, 0);
	for (unsigned int i = 0; i < SIZE; ++i) {
		training_indexes[i] = i;
	}

	double threshold = 0.0;
	vector<pair_similarity> neighborhood_user;
	vector<unsigned int> pair_indexes_user;

	vector<pair_similarity> neighborhood_item;
	vector<unsigned int> pair_indexes_item;

	if (MF_SIMILARITY_USER) {

		int count = 0;
		for (unsigned int u = 0; u < users.size(); ++u) {
			for (unsigned v = 0; v < users.size(); ++v) {
				if (u != v) {
					double value = calc_similarity_user(u, v,
							MF_SIMILARITY_USER);
					if (abs(value) > threshold) {
						pair_similarity pair;
						pair.u = u;
						pair.v = v;
						pair.s = value;
						//pair.weight = calc_similarity_user(u, v, INTERSECTION) + 1;

						pair.weight = 1;

						neighborhood_user.push_back(pair);
						pair_indexes_user.push_back(count++);
					}

				}

			}
		}
	}

	if (MF_SIMILARITY_ITEM) {
		//cout << "ITEM" << endl;
		int count = 0;
		for (unsigned int u = 0; u < items.size(); ++u) {
			for (unsigned v = 0; v < items.size(); ++v) {

				if (u != v) {
					double value = calc_similarity_item(u, v,
							MF_SIMILARITY_ITEM);
					if (abs(value) > threshold) {
						pair_similarity pair;
						pair.u = u;
						pair.v = v;
						pair.s = value;
						//pair.weight = calc_similarity_item(u, v, INTERSECTION) + 1;
						pair.weight = 1;

						neighborhood_item.push_back(pair);
						pair_indexes_item.push_back(count++);
					}

				}

			}
		}
		//cin.get();
	}

	for (unsigned int it = 0; it < MF_NUM_ITERATIONS; ++it) {
		cout << it << endl;
		random_shuffle(training_indexes.begin(), training_indexes.end());

		for (unsigned int i = 0; i < SIZE; ++i) {
			Vote *v = trainingset[training_indexes[i]];

			unsigned int u_id = v->userId;
			unsigned int i_id = v->itemId;

			double r = v->rating;
			double err = (r - dot_product(p[u_id], q[i_id]));

			if (MF_NORMALIZE) {
				r = (v->rating - MIN_RATING) / DELTA_RATING;
				double sig_p = sigmoid(dot_product(p[u_id], q[i_id]));
				err = (r - sig_p) * sig_p * (1 - sig_p);
			}

			for (unsigned int k = 0; k < MF_NUM_FACTORS; ++k) {
				p[u_id][k] += MF_ALPHA
						* (err * q[i_id][k] - MF_LAMBDA * p[u_id][k]);
				q[i_id][k] += MF_ALPHA
						* (err * p[u_id][k] - MF_LAMBDA * q[i_id][k]);
			}
		}

		//TODO: asymmetric functionality must be implemented here
		if (MF_SIMILARITY_USER) {

			random_shuffle(pair_indexes_user.begin(), pair_indexes_user.end());
			for (unsigned int i = 0; i < pair_indexes_user.size(); ++i) {

				unsigned int idx = pair_indexes_user[i];

				const unsigned int u = neighborhood_user[idx].u;
				const unsigned int v = neighborhood_user[idx].v;
				const double s = neighborhood_user[idx].s;
				const double w = neighborhood_user[idx].weight;

				double sig_p = dot_product(p[u], p[v]);
				double err = (s - sig_p);
				if (MF_NORMALIZE) {
					sig_p = sigmoid(dot_product(p[u], p[v]));
					err = (s - sig_p) * sig_p * (1 - sig_p);
				}

				for (unsigned int k = 0; k < MF_NUM_FACTORS; ++k) {
					p[u][k] += MF_ALPHA * (w * err * p[v][k]);
					p[v][k] += MF_ALPHA * (w * err * p[u][k]);
				}
			}
		}

		if (MF_SIMILARITY_ITEM) {

			random_shuffle(pair_indexes_item.begin(), pair_indexes_item.end());
			for (unsigned int i = 0; i < pair_indexes_item.size(); ++i) {

				unsigned int idx = pair_indexes_item[i];
				const unsigned int u = neighborhood_item[idx].u;
				const unsigned int v = neighborhood_item[idx].v;
				const double s = neighborhood_item[idx].s;
				const double w = neighborhood_item[idx].weight;

				double sig_p = dot_product(q[u], q[v]);
				double err = (s - sig_p);
				if (MF_NORMALIZE) {
					sig_p = sigmoid(dot_product(q[u], q[v]));
					err = (s - sig_p) * sig_p * (1 - sig_p);
				}

				for (unsigned int k = 0; k < MF_NUM_FACTORS; ++k) {
					q[u][k] += MF_ALPHA * (w * err * q[v][k]);
					q[v][k] += MF_ALPHA * (w * err * q[u][k]);
				}
			}
		}
	}
}

void run_matrix_factorization(vector<TestItem> &test,
		vector<Vote *> &trainingset, const unsigned int fold) {

	std::normal_distribution<double> distribution(0, 0.1);

	vector<vector<double> > p(users.size(),
			vector<double>(MF_NUM_FACTORS, 0.0));
	for (unsigned int i = 0; i < users.size(); ++i) {
		for (unsigned int k = 0; k < MF_NUM_FACTORS; ++k) {
			p[i][k] = distribution(generator);
		}
	}

	vector<vector<double> > q(items.size(),
			vector<double>(MF_NUM_FACTORS, 0.0));
	for (unsigned int i = 0; i < items.size(); ++i) {
		for (unsigned int k = 0; k < MF_NUM_FACTORS; ++k) {
			q[i][k] = distribution(generator);
		}
	}

	string output_file;
	switch (MF_ALGORITHM) {
	case 4: {

		sgd_smf(trainingset, p, q);

		char buffer[250];
		sprintf(buffer, "%u-%u-%u-%u-%3.5f-%3.5f", MF_NUM_FACTORS,
				MF_SIMILARITY_USER, MF_SIMILARITY_ITEM, MF_NORMALIZE, MF_LAMBDA,
				MF_ALPHA);

		output_file = string(buffer);
		break;
	}

	}

	Experiment exp;
	unsigned int num_users = 0;

	for (unsigned int i = 0; i < test.size(); ++i) {
		User *u = test[i].user;

		set<unsigned int> liked;
		set<unsigned int> not_liked;

		for (size_t j = 0; j < test[i].reviews.size(); ++j) {
			unsigned int id = itemIds[test[i].reviews[j]->itemId];

			if (cmp(test[i].reviews[j]->rating, u->avg) >= 0) {
				liked.insert(id);
			} else {
				not_liked.insert(id);
			}
		}
		if (liked.size() < 1) {
			continue;
		}

		++num_users;

		vector<double> rank(items.size(), 0);
		for (unsigned int j = 0; j < items.size(); ++j) {
			rank[j] = dot_product(p[u->id], q[j]);
		}

		string result = doExperimentAll(u, test[i].reviews, rank, liked,
				not_liked, exp);
		//		save_user_result(MF_SIMILARITY_USER, MF_SIMILARITY_ITEM, MF_NORMALIZE,
		//				result);

		if (fold == 0) {
			ofstream outfile;
			string file = std::to_string(fold) + "-" + output_file
					+ ".breakdown";

			outfile.open(file.c_str(), std::ios_base::app);
			outfile << result << endl;
			outfile.close();
		}
	}

	exp.normalize(num_users);

	print_result(exp, output_file + ".out");

}
// ######### END MF

// ######### BEGIN READING

void generate_dataset() {

	for (unsigned int k = BEGIN_K_FOLD; k < K_FOLD; ++k) {

		ofstream train_f;
		ofstream test_f;

		string f1 = to_string(k) + ".csv";
		string f2 = to_string(k + K_FOLD) + ".csv";
		train_f.open(f1.c_str(), std::ios_base::app);
		test_f.open(f2.c_str(), std::ios_base::app);

		for (size_t z = 0; z < reviews.size(); ++z) {
			Review* r = reviews[z];

			if (K_FOLD > 1 && r->fold == k) {
				test_f << r->userId << "," << r->itemId << "," << r->rating
						<< endl;
			} else {
				train_f << r->userId << "," << r->itemId << "," << r->rating
						<< endl;
			}
		}

		train_f.close();
		test_f.close();
	}

}

void read_data(const char* filename) {
	Gen generator;

	ifstream file(filename);
	string line;

	unsigned int userId;
	unsigned int itemId;
	float rating;

	getline(file, line); // reading header

	MAX_RATING = INT_MIN;
	MIN_RATING = INT_MAX;

	set<float> ratings;

	while (getline(file, line)) {
		stringstream ss(line);
		string tok;

		getline(ss, tok, DELIM);
		userId = atoi(tok.c_str());

		getline(ss, tok, DELIM);
		itemId = atoi(tok.c_str());

		getline(ss, tok, DELIM);
		rating = atof(tok.c_str());

		if (rating <= 0)
			continue;

		if (rating > MAX_RATING) {
			MAX_RATING = rating;
		} else if (rating < MIN_RATING) {
			MIN_RATING = rating;
		}

//timestamp
//getline(ss, tok, DELIM);

		Review * r = new Review();
		r->userId = userId;
		r->itemId = itemId;
		r->rating = rating;
		r->fold = generator.next();

		k_count[r->fold]++;

		ratings.insert(r->rating);

		reviews.push_back(r);
	}
	DELTA_RATING = MAX_RATING - MIN_RATING;

// levels for ordinal gibbs
	vector<double> sorted_ratings;
	for (set<float>::iterator it = ratings.begin(); it != ratings.end(); ++it) {
		sorted_ratings.push_back(*it);
	}
	sort(sorted_ratings.begin(), sorted_ratings.end());

}

void clear() {
	userIds.clear();
	rUserIds.clear();

	itemIds.clear();
	rItemIds.clear();

	for (size_t i = 0; i < users.size(); ++i) {
		delete users[i];
	}
	users.clear();

	for (size_t i = 0; i < items.size(); ++i)
		delete items[i];
	items.clear();

}

void calcAdamicAdar() {
	for (unsigned j = 0; j < users.size(); ++j) {
		User *u = users[j];

		for (unsigned k = 0; k < u->ratings.size(); ++k) {
			u->adamic_adar_sum_neighbors +=
					items[u->ratings[k]->itemId]->adamic_adar;
		}
	}

	for (unsigned j = 0; j < items.size(); ++j) {
		Item *i = items[j];
		for (unsigned k = 0; k < i->ratings.size(); ++k) {
			i->adamic_adar_sum_neighbors +=
					users[i->ratings[k]->userId]->adamic_adar;
		}
	}
}

// scales user rating so that 1 if above user average or 0 otherwise
void scaleRating(User *u) {
	u->avg = 0;
	u->std = 0;
	u->adamic_adar = 1 / log(u->ratings.size());

	for (size_t i = 0; i < u->ratings.size(); ++i) {
		u->avg += u->ratings[i]->rating;
	}
	u->avg /= u->ratings.size();

	for (size_t i = 0; i < u->ratings.size(); ++i) {

		Item *p = items[u->ratings[i]->itemId];
		p->nRatings++;

		u->std += pow(u->ratings[i]->rating - u->avg, 2);

		if (cmp(u->ratings[i]->rating, u->avg) >= 0) {
			p->nUps++;
		}
	}
	u->std = sqrt(u->std / (u->ratings.size() - 1));
}

void scaleItem(Item *u) {
	u->avg = 0;
	u->adamic_adar = 1 / log(u->ratings.size());

	for (size_t i = 0; i < u->ratings.size(); ++i) {
		u->avg += u->ratings[i]->rating;
	}
	u->avg /= u->ratings.size();
}

void kfold(char algorithm) {

	for (unsigned int k = BEGIN_K_FOLD; k < END_K_FOLD; ++k) {

		map<unsigned int, vector<Review*> > test; // key is  real user id in dataset
		map<unsigned int, unsigned int>::iterator it;
		vector<TestItem> testset;
		vector<Vote *> trainingset;

		unsigned int num_feedbacks = 0;

		for (size_t z = 0; z < reviews.size(); ++z) {
			Review* r = reviews[z];

			if (K_FOLD > 1 && r->fold == k) {
				test[r->userId].push_back(r);
			} else {

				++num_feedbacks;

				unsigned int userId = r->userId;
				unsigned int itemId = r->itemId;

				User * u;
				Item *p;

				it = userIds.find(userId);
				if (it == userIds.end()) {
					int id = users.size();
					userIds[userId] = id;
					rUserIds[id] = userId;
					u = new User();
					u->id = id;
					users.push_back(u);
				} else {
					u = users[it->second];
				}

				it = itemIds.find(itemId);
				if (it == itemIds.end()) {
					int id = items.size();
					itemIds[itemId] = id;
					rItemIds[id] = itemId;
					p = new Item();
					p->id = id;
					items.push_back(p);
				} else {
					p = items[it->second];
				}
				p->users.insert(u->id);
				u->items.insert(p->id);

				Vote *v = new Vote();
				v->itemId = p->id;
				v->userId = u->id;
				v->rating = r->rating;

				u->ratings.push_back(v);
				p->ratings.push_back(v);

				trainingset.push_back(v);
			}
		}

		for (size_t i = 0; i < users.size(); ++i) {
			scaleRating(users[i]);
			sort(users[i]->ratings.begin(), users[i]->ratings.end(),
					voteComparator);
		}

		for (size_t i = 0; i < items.size(); ++i) {
			Item *p = items[i];
			scaleItem(p);

			p->posteriori_success = (BETA_alpha + p->nUps)
					/ (BETA_alpha + BETA_beta + p->nRatings);
			p->odds_ratio = p->posteriori_success / (1 - p->posteriori_success);

			sort(p->ratings.begin(), p->ratings.end(), voteComparator2);
		}

		calcAdamicAdar();

		for (map<unsigned int, vector<Review*> >::iterator it2 = test.begin();
				it2 != test.end(); ++it2) {

			if (userIds.find(it2->first) != userIds.end()) {
				TestItem unit;
				unit.user = users[userIds[it2->first]];
				unit.reviews = it2->second;
				testset.push_back(unit);
			}
		}

		Experiment result;
		switch (algorithm) {
		case MATRIX_FACTORIZATION: {
			run_matrix_factorization(testset, trainingset, k);
			break;
		}
		}
		clear();
	}
}

int main(int argc, char **argv) {
	srand(0);

	topN.push_back(5);
	topN.push_back(10);

	NUM_THREADS = atoi(argv[1]);
	char* filename = argv[2];
	K_FOLD = atoi(argv[3]);
	BEGIN_K_FOLD = atoi(argv[4]);
	END_K_FOLD = atoi(argv[5]);
	int algorithm = atoi(argv[6]);

	k_count.resize(K_FOLD, 0);
	read_data(filename);

	switch (algorithm) {
	case 0:
		generate_dataset();
		break;
	case MATRIX_FACTORIZATION: {
		MF_ALGORITHM = atoi(argv[7]);
		MF_NUM_FACTORS = atoi(argv[8]);
		MF_SIMILARITY_USER = atoi(argv[9]);
		MF_SIMILARITY_ITEM = atoi(argv[10]);
		MF_NORMALIZE = atoi(argv[11]);

		MF_NUM_ITERATIONS = atoi(argv[12]);

		MF_LAMBDA = atof(argv[13]);
		MF_ALPHA = atof(argv[14]);

		break;
	}

	default:
		cout << "error" << endl;
		return 0;
	}

	kfold(algorithm);
	return 0;
}