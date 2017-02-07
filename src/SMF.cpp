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
#include <cfloat>
#include <ctime>
#include <bitset>

#include <omp.h>

#define DELIM ','
#define EPS 1e-8

#define BETA_alpha 1.0
#define BETA_beta 1.0

#define MATRIX_FACTORIZATION 13

#define NONE_SIMILARITY 0

#define PEARSON_SIMILARITY 1
#define JACCARD_SIMILARITY 2
#define COSINE_SIMILARITY 3
#define LIANG_SIMILARITY 4

#define ASYMMETRIC_JACCARD_SIMILARITY 11
#define ASYMMETRIC_ADAMIC_ADAR_SIMILARIY 12
#define ACOS 13
#define AMSD 14

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
			id(0), avg(0), posteriori_success(0), odds_ratio(0), nUps(1), nRatings(
					2), adamic_adar(0), adamic_adar_sum_neighbors(0), liang_sum(
					0) {
	}
	~Item() {
		users.clear();
		ratings.clear();
	}

	unsigned int id;
	double avg;
	double posteriori_success;
	double odds_ratio;
	unsigned int nUps;
	unsigned int nRatings;
	double adamic_adar;
	double adamic_adar_sum_neighbors;
	double liang_sum;
	set<unsigned int> users;
	vector<Vote *> ratings;
};
typedef struct Item Item;

struct User {
	unsigned int id;
	double avg;
	double std;
	double adamic_adar;
	double adamic_adar_sum_neighbors;
	double liang_sum;

	set<unsigned int> items;
	vector<Vote *> ratings;

	User() :
			id(0), avg(0), std(0), adamic_adar(0), adamic_adar_sum_neighbors(0), liang_sum(
					0) {
	}
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

map<unsigned int, vector<unsigned> > social_network;

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

inline unsigned get_index(unsigned i, unsigned j, unsigned n) {
	return n * i + j - 0.5 * ((i + 2) * (i + 1));
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

vector<double> user_similarity_matrix;
vector<double> item_similarity_matrix;

// item similarity

double calc_similarity_item(unsigned int p, unsigned int q, unsigned int type) {

	vector<Vote *> u = items[p]->ratings;
	vector<Vote *> v = items[q]->ratings;

	vector<float> p_v;
	vector<float> q_v;

	vector<unsigned> intersection;

	{
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

	case ASYMMETRIC_JACCARD_SIMILARITY: {
		value = ((double) p_v.size());
		break;
	}
	case ASYMMETRIC_ADAMIC_ADAR_SIMILARIY: {
		value = 0;
		for (unsigned i = 0; i < intersection.size(); ++i) {
			value += users[intersection[i]]->adamic_adar;
		}

//		value /= items[p]->adamic_adar_sum_neighbors;

		break;
	}
	case INTERSECTION: {
		value = p_v.size();
		break;
	}
	case ACOS: {

		//eq. 1
		value = ((double) p_v.size()) / u.size();

		//eq. 2
		value *= (2.0 * p_v.size()) / (u.size() + v.size());

		//cosine
		double cosine;
		double num = 0;
		double x_den = vct_norm(p_v);
		double y_den = vct_norm(q_v);
		for (unsigned int i = 0; i < p_v.size(); ++i) {
			num += p_v[i] * q_v[i];
		}
		cosine = (num / sqrt(x_den * y_den));

		//eq. 3
		value *= cosine;

		break;
	}
	case AMSD: {

		//eq. 1
		value = ((double) p_v.size()) / u.size();

		//eq. 2
		value *= (2.0 * p_v.size()) / (u.size() + v.size());

		//eq. 3

		double num = 0;
		double MSD;
		for (unsigned int i = 0; i < p_v.size(); ++i) {
			num += pow((p_v[i] - q_v[i]), 2.0);
		}
		MSD = (num / (u.size() + v.size() - p_v.size()));

		//eq. 5

		double L = DELTA_RATING * DELTA_RATING;
		double sim_u_v = (L - MSD) / L;

		//eq. 6

		value *= sim_u_v;

		break;
	}

	}

	return value;
}

double LIANG_SUM_ITEM_SIMILARITY = 0;
double LIANG_SUM_USER_SIMILARITY = 0;

void calc_liang_item_similarity() {
	for (unsigned i = 0; i < items.size(); ++i) {
		for (unsigned j = i + 1; j < items.size(); ++j) {

			unsigned int index = get_index(i, j, items.size());

			double value = calc_similarity_item(i, j, INTERSECTION);
			item_similarity_matrix[index] = value;
			items[i]->liang_sum += value;
			items[j]->liang_sum += value;
			LIANG_SUM_ITEM_SIMILARITY += value;
		}
	}
}

double get_liang_item_similarity(unsigned i, unsigned j) {
	if (i > j) {
		int aux = i;
		i = j;
		j = aux;
	}

	unsigned int index = get_index(i, j, items.size());

	double value = (item_similarity_matrix[index] * LIANG_SUM_ITEM_SIMILARITY)
			/ (items[i]->liang_sum * items[j]->liang_sum);
	value = log(value);
	value = value > 0 ? value : 0;

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
	unsigned int index = get_index(p, q, items.size());

	switch (type) {
	case LIANG_SIMILARITY:
		return get_liang_item_similarity(p, q);
		break;
	default:
		if (item_similarity_matrix[index] < 0) {
			item_similarity_matrix[index] = calc_similarity_item(p, q, type);
		}
		break;
	}

	return item_similarity_matrix[index];

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

	case ASYMMETRIC_JACCARD_SIMILARITY: {
		value = ((double) x_v.size());
		break;
	}
	case ASYMMETRIC_ADAMIC_ADAR_SIMILARIY: {
		value = 0.0;
		for (unsigned i = 0; i < intersection.size(); ++i) {
			value += items[intersection[i]]->adamic_adar;
		}
		break;
	}

	case INTERSECTION: {
		value = x_v.size();
		break;

	}
	case ACOS: {

		//eq. 1
		value = ((double) x_v.size()) / u->ratings.size();

		//eq. 2
		value *= (2.0 * x_v.size()) / (u->ratings.size() + v->ratings.size());

		//cosine
		double cosine;
		double num = 0;
		double x_den = vct_norm(x_v);
		double y_den = vct_norm(y_v);
		for (unsigned int i = 0; i < x_v.size(); ++i) {
			num += x_v[i] * y_v[i];
		}
		cosine = (num / sqrt(x_den * y_den));

		//eq. 3
		value *= cosine;

		break;
	}
	case AMSD: {

		//eq. 1
		value = ((double) x_v.size()) / u->ratings.size();

		//eq. 2
		value *= (2.0 * x_v.size()) / (u->ratings.size() + v->ratings.size());

		//eq. 4

		double num = 0;
		double MSD;

		for (unsigned int i = 0; i < x_v.size(); ++i) {
			num += pow((x_v[i] - y_v[i]), 2.0);
		}
		MSD = (num / (u->ratings.size() + v->ratings.size() - x_v.size()));

		//eq. 5
		double L = DELTA_RATING * DELTA_RATING;
		double sim_u_v = (L - MSD) / L;

		//eq. 6
		value *= sim_u_v;
		break;
	}
	}

	return value;
}

void calc_liang_user_similarity() {
	for (unsigned i = 0; i < users.size(); ++i) {
		for (unsigned j = i + 1; j < users.size(); ++j) {

			unsigned int index = get_index(i, j, users.size());

			double value = calc_similarity_user(i, j, INTERSECTION);
			user_similarity_matrix[index] = value;
			users[i]->liang_sum += value;
			users[j]->liang_sum += value;
			LIANG_SUM_USER_SIMILARITY += value;
		}
	}
}

double get_liang_user_similarity(unsigned i, unsigned j) {
	if (i > j) {
		int aux = i;
		i = j;
		j = aux;
	}

	unsigned int index = get_index(i, j, users.size());

	double value = (user_similarity_matrix[index] * LIANG_SUM_USER_SIMILARITY)
			/ (users[i]->liang_sum * users[j]->liang_sum);
	value = log(value);

	value = value > 0 ? value : 0;

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

	if (user_similarity_matrix[index] < 0) {
		user_similarity_matrix[index] = calc_similarity_user(p, q, type);
	}

	return user_similarity_matrix[index];
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

void sgd_social_MF(const vector<Vote *> &trainingset,
		map<unsigned int, vector<unsigned> > &user_neighbors,
		vector<vector<double> > &p, vector<vector<double> > &q) {

	const unsigned int SIZE = trainingset.size();
	const unsigned int SOCIAL_SIZE = user_neighbors.size();

	vector<unsigned int> training_indexes(SIZE, 0);
	for (unsigned int i = 0; i < SIZE; ++i) {
		training_indexes[i] = i;
	}

	vector<unsigned int> social_indexes;
	for (map<unsigned int, vector<unsigned> >::iterator it2 =
			user_neighbors.begin(); it2 != user_neighbors.end(); ++it2) {
		unsigned u = it2->first;
//		cout << u << " " << it2->second.size() << endl;
		social_indexes.push_back(u);
	}
//	cin.get();

	for (unsigned int it = 0; it < MF_NUM_ITERATIONS; ++it) {
		random_shuffle(training_indexes.begin(), training_indexes.end());

		random_shuffle(social_indexes.begin(), social_indexes.end());

		// prediction error
#pragma omp parallel num_threads(NUM_THREADS)
		{

#pragma omp for
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
		}

#pragma omp parallel num_threads(NUM_THREADS)
		{
#pragma omp for
			for (unsigned int a = 0; a < SOCIAL_SIZE; ++a) {

				unsigned u = social_indexes[a];
				unsigned n_neigh_u = user_neighbors[u].size();

				vector<double> sum_neighbors(MF_NUM_FACTORS, 0.0);
				vector<double> sum_neighbors2(MF_NUM_FACTORS, 0.0);

				for (unsigned i = 0; i < n_neigh_u; ++i) {
					unsigned v = user_neighbors[u][i];
					unsigned n_neigh_v = user_neighbors[v].size();

					for (unsigned f = 0; f < MF_NUM_FACTORS; ++f) {
						sum_neighbors[f] += p[v][f];

						double aux = p[v][f];
						for (unsigned j = 0; j < n_neigh_v; ++j) {
							unsigned x = user_neighbors[v][j];
							aux -= p[x][f] / n_neigh_v;
						}
						aux = aux / n_neigh_v;

						sum_neighbors2[f] += aux;
					}
				}

				for (unsigned f = 0; f < MF_NUM_FACTORS; ++f) {
					p[u][f] += -MF_ALPHA
							* (p[u][f] - sum_neighbors[f] / n_neigh_u
									- sum_neighbors2[f]);
				}
			}
		}
	}
}

void smf(const vector<Vote *> &trainingset, vector<vector<double> > &p,
		vector<vector<double> > &q) {

	double threshold = 0.0;

	vector<vector<pair_similarity>> neighborhood_user(users.size());

	vector<vector<pair_similarity>> neighborhood_item(items.size());

	vector<vector<double> > p_gradient(users.size(),
			vector<double>(MF_NUM_FACTORS, 0.0));
	vector<vector<double> > q_gradient(items.size(),
			vector<double>(MF_NUM_FACTORS, 0.0));

// calcula a similaridade dos usuarios
	if (MF_SIMILARITY_USER) {

		int count = 0;
		for (unsigned int u = 0; u < users.size(); ++u) {
			for (unsigned v = u + 1; v < users.size(); ++v) {
				double value = 0;

				switch (MF_SIMILARITY_USER) {
				case LIANG_SIMILARITY:
					value = get_liang_user_similarity(u, v);
					break;
				default:
					value = calc_similarity_user(u, v, MF_SIMILARITY_USER);
					break;
				}

				if (abs(value) > threshold) {
					pair_similarity pair;
					pair.u = u;
					pair.v = v;
					pair.s = value;
					//pair.weight = calc_similarity_user(u, v, INTERSECTION) + 1;

					pair.weight = 1;

					neighborhood_user[u].push_back(pair);
				}
			}
		}
	}

//calcula a similaridade dos itens
	if (MF_SIMILARITY_ITEM) {
		int count = 0;
		for (unsigned int u = 0; u < items.size(); ++u) {
			for (unsigned v = u + 1; v < items.size(); ++v) {

				double value = 0;

				switch (MF_SIMILARITY_ITEM) {
				case LIANG_SIMILARITY:
					value = get_liang_item_similarity(u, v);
					break;
				default:
					value = calc_similarity_item(u, v, MF_SIMILARITY_ITEM);
					break;
				}

				if (abs(value) > threshold) {
					pair_similarity pair;
					pair.u = u;
					pair.v = v;
					pair.s = value;
					//pair.weight = calc_similarity_item(u, v, INTERSECTION) + 1;
					pair.weight = 1;

					neighborhood_item[u].push_back(pair);
				}
			}
		}
	}

	for (unsigned int it = 0; it < MF_NUM_ITERATIONS; ++it) {

		for (unsigned i = 0; i < users.size(); ++i) {

			User *u = users[i];
			for (unsigned j = 0; j < u->ratings.size(); ++j) {
				Vote *v = u->ratings[j];

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
					p_gradient[u_id][k] += -err * q[i_id][k];
					q_gradient[i_id][k] += -err * p[u_id][k];
				}
			}
		}

		if (MF_SIMILARITY_USER) {

			for (unsigned int i = 0; i < users.size(); ++i) {
				for (unsigned j = 0; j < neighborhood_user[i].size(); ++j) {

					pair_similarity pair_sim = neighborhood_user[i][j];

					const unsigned int u = pair_sim.u;
					const unsigned int v = pair_sim.v;
					const double s = pair_sim.s;
					const double w = pair_sim.weight;

					double sig_p = dot_product(p[u], p[v]);
					double err = (s - sig_p); // s eh similaridade sig_p produdto escalar pu pv
					if (MF_NORMALIZE) {
						sig_p = sigmoid(dot_product(p[u], p[v]));
						err = (s - sig_p) * sig_p * (1 - sig_p);
					}

					for (unsigned int k = 0; k < MF_NUM_FACTORS; ++k) {
						p_gradient[u][k] += -err * p[v][k];
						p_gradient[v][k] += -err * p[u][k];
					}
				}
			}

			if (MF_SIMILARITY_ITEM) {

				for (unsigned int i = 0; i < items.size(); ++i) {
					for (unsigned j = 0; j < neighborhood_item[i].size(); ++j) {

						pair_similarity pair_sim = neighborhood_item[i][j];

						const unsigned int u = pair_sim.u;
						const unsigned int v = pair_sim.v;
						const double s = pair_sim.s;
						const double w = pair_sim.weight;

						double sig_p = dot_product(q[u], q[v]);
						double err = (s - sig_p);
						if (MF_NORMALIZE) {
							sig_p = sigmoid(dot_product(q[u], q[v]));
							err = (s - sig_p) * sig_p * (1 - sig_p);
						}

						for (unsigned int k = 0; k < MF_NUM_FACTORS; ++k) {
							q_gradient[u][k] += -err * q[v][k];
							q_gradient[v][k] += -err * q[u][k];
						}
					}
				}
			}

			for (unsigned int i = 0; i < users.size(); ++i) {
				for (unsigned int k = 0; k < MF_NUM_FACTORS; ++k) {
					p[i][k] -= MF_ALPHA
							* (p_gradient[i][k] + MF_LAMBDA * p[i][k]);

					p_gradient[i][k] = 0;
				}
			}

			for (unsigned int i = 0; i < items.size(); ++i) {
				for (unsigned int k = 0; k < MF_NUM_FACTORS; ++k) {
					q[i][k] -= MF_ALPHA
							* (q_gradient[i][k] + MF_LAMBDA * q[i][k]);

					q_gradient[i][k] = 0;
				}
			}
		}
	}
}

void sgd_smf(const vector<Vote *> &trainingset, vector<vector<double> > &p,
		vector<vector<double> > &q) {

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

// calcula a similaridade dos usuarios
	if (MF_SIMILARITY_USER) {

		int count = 0;
		for (unsigned int u = 0; u < users.size(); ++u) {
			for (unsigned v = u + 1; v < users.size(); ++v) {
				double value = 0;

				switch (MF_SIMILARITY_USER) {
				case LIANG_SIMILARITY:
					value = get_liang_user_similarity(u, v);
					break;
				default:
					value = calc_similarity_user(u, v, MF_SIMILARITY_USER);
					break;
				}

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

//calcula a similaridade dos itens
	if (MF_SIMILARITY_ITEM) {
		int count = 0;
		for (unsigned int u = 0; u < items.size(); ++u) {
			for (unsigned v = u + 1; v < items.size(); ++v) {

				double value = 0;

				switch (MF_SIMILARITY_ITEM) {
				case LIANG_SIMILARITY:
					value = get_liang_item_similarity(u, v);
					break;
				default:
					value = calc_similarity_item(u, v, MF_SIMILARITY_ITEM);
					break;
				}

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

	for (unsigned int it = 0; it < MF_NUM_ITERATIONS; ++it) {
		random_shuffle(training_indexes.begin(), training_indexes.end());

#pragma omp parallel num_threads(NUM_THREADS)
		{

#pragma omp for

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
					double temp = p[u_id][k];
					p[u_id][k] += MF_ALPHA
							* (err * q[i_id][k] - MF_LAMBDA * p[u_id][k]);
					q[i_id][k] += MF_ALPHA
							* (err * temp - MF_LAMBDA * q[i_id][k]);
				}
			}
		}

		if (MF_SIMILARITY_USER) {
			random_shuffle(pair_indexes_user.begin(), pair_indexes_user.end());

#pragma omp parallel num_threads(NUM_THREADS)
			{

#pragma omp for
				for (unsigned int i = 0; i < pair_indexes_user.size(); ++i) {

					unsigned int idx = pair_indexes_user[i];

					const unsigned int u = neighborhood_user[idx].u;
					const unsigned int v = neighborhood_user[idx].v;
					const double s = neighborhood_user[idx].s;
					const double w = neighborhood_user[idx].weight;

					double sig_p = dot_product(p[u], p[v]);
					double err = (s - sig_p); // s eh similaridade sig_p produdto escalar pu pv
					if (MF_NORMALIZE) {
						sig_p = sigmoid(dot_product(p[u], p[v]));
						err = (s - sig_p) * sig_p * (1 - sig_p);
					}

					for (unsigned int k = 0; k < MF_NUM_FACTORS; ++k) {

						double temp = p[u][k];
						p[u][k] += MF_ALPHA * (w * err * p[v][k]);
						p[v][k] += MF_ALPHA * (w * err * temp);
					}
				}
			}
		}

		if (MF_SIMILARITY_ITEM) {
			random_shuffle(pair_indexes_item.begin(), pair_indexes_item.end());

#pragma omp parallel num_threads(NUM_THREADS)
			{

#pragma omp for
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
						double temp = q[u][k];
						q[u][k] += MF_ALPHA * (w * err * q[v][k]);
						q[v][k] += MF_ALPHA * (w * err * temp);
					}
				}
			}
		}

//		{
//			double global_error = 0;
//			for (unsigned int i = 0; i < SIZE; ++i) {
//				Vote *v = trainingset[training_indexes[i]];
//
//				unsigned int u_id = v->userId;
//				unsigned int i_id = v->itemId;
//
//				double r = v->rating;
//				double err = (r - dot_product(p[u_id], q[i_id]));
//
//				global_error += pow(err,2);
//			}
//
//			global_error = sqrt(global_error/SIZE);
//			cout << global_error << endl;
//		}
//		cout << sqrt(monitor_err) << endl;
//		for (unsigned int k = 0; k < MF_NUM_FACTORS; ++k) {
//			cout << p[0][k] << ",";
//		}
//		cout << endl;
	}
}

void sgd_smf_social_similarity(const vector<Vote *> &trainingset,
		vector<vector<double> > &p, vector<vector<double> > &q,
		map<unsigned int, vector<unsigned> > user_neighbors) {

	const unsigned int SIZE = trainingset.size();

	vector<unsigned int> training_indexes(SIZE, 0);
	for (unsigned int i = 0; i < SIZE; ++i) {
		training_indexes[i] = i;
	}

	double threshold = 0.0;
	vector<pair_similarity> neighborhood_user;
	vector<unsigned int> pair_indexes_user;

// calcula a similaridade dos usuarios
	set<pair<unsigned, unsigned>> pairs;

	for (map<unsigned int, vector<unsigned> >::iterator it2 =
			user_neighbors.begin(); it2 != user_neighbors.end(); ++it2) {
		unsigned u = it2->first;

		vector<unsigned> neigh = it2->second;

		for (unsigned j = 0; j < it2->second.size(); ++j) {
			unsigned v = it2->second[j];

			if (v > u) {
				unsigned aux = v;
				v = u;
				u = aux;
			}

			pairs.insert(make_pair(u, v));
		}
	}

	int count = 0;
	for (set<pair<unsigned, unsigned>>::iterator it = pairs.begin();
			it != pairs.end(); ++it) {
		unsigned u = it->first;
		unsigned v = it->second;

		double value = 0;

		switch (MF_SIMILARITY_USER) {
		case LIANG_SIMILARITY:
			value = get_liang_user_similarity(u, v);
			break;
		default:
			value = calc_similarity_user(u, v, MF_SIMILARITY_USER);
			break;
		}

		if (abs(value) > threshold) {
			pair_similarity p_sim;
			p_sim.u = u;
			p_sim.v = v;
			p_sim.s = value;
			//pair.weight = calc_similarity_user(u, v, INTERSECTION) + 1;

			p_sim.weight = 1;

			neighborhood_user.push_back(p_sim);
			pair_indexes_user.push_back(count++);
		}
	}

	for (unsigned int it = 0; it < MF_NUM_ITERATIONS; ++it) {
		random_shuffle(training_indexes.begin(), training_indexes.end());

#pragma omp parallel num_threads(NUM_THREADS)
		{

#pragma omp for

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
					double temp = p[u_id][k];
					p[u_id][k] += MF_ALPHA
							* (err * q[i_id][k] - MF_LAMBDA * p[u_id][k]);
					q[i_id][k] += MF_ALPHA
							* (err * temp - MF_LAMBDA * q[i_id][k]);
				}
			}
		}

		if (MF_SIMILARITY_USER) {
			random_shuffle(pair_indexes_user.begin(), pair_indexes_user.end());

#pragma omp parallel num_threads(NUM_THREADS)
			{

#pragma omp for
				for (unsigned int i = 0; i < pair_indexes_user.size(); ++i) {

					unsigned int idx = pair_indexes_user[i];

					const unsigned int u = neighborhood_user[idx].u;
					const unsigned int v = neighborhood_user[idx].v;
					const double s = neighborhood_user[idx].s;
					const double w = neighborhood_user[idx].weight;

					double sig_p = dot_product(p[u], p[v]);
					double err = (s - sig_p); // s eh similaridade sig_p produdto escalar pu pv
					if (MF_NORMALIZE) {
						sig_p = sigmoid(dot_product(p[u], p[v]));
						err = (s - sig_p) * sig_p * (1 - sig_p);
					}

					for (unsigned int k = 0; k < MF_NUM_FACTORS; ++k) {
						double temp = p[u][k];
						p[u][k] += MF_ALPHA * (w * err * p[v][k]);
						p[v][k] += MF_ALPHA * (w * err * temp);
					}
				}
			}
		}
	}
}

void als_smf_new_model(const vector<Vote *> &trainingset,
		vector<vector<double> > &p, vector<vector<double> > &q) {

	double threshold = 0.0;

	vector<vector<pair_similarity*>> neighborhood_user(users.size());

	vector<vector<pair_similarity*>> neighborhood_item(items.size());

// calcula a similaridade dos usuarios
	if (MF_SIMILARITY_USER) {
		unsigned count = 0;

		for (unsigned int u = 0; u < users.size(); ++u) {
			for (unsigned v = u + 1; v < users.size(); ++v) {
				double value = 0;

				switch (MF_SIMILARITY_USER) {
				case LIANG_SIMILARITY:
					value = get_liang_user_similarity(u, v);
					break;
				default:
					value = calc_similarity_user(u, v, MF_SIMILARITY_USER);
					break;
				}

				if (abs(value) > threshold) {
					pair_similarity *pair = new pair_similarity();
					pair->u = u;
					pair->v = v;
					pair->s = value;
					//pair.weight = calc_similarity_user(u, v, INTERSECTION) + 1;
					pair->weight = 1;
					neighborhood_user[u].push_back(pair);
					neighborhood_user[v].push_back(pair);
					++count;
				}
			}
		}
	}

//calcula a similaridade dos itens
	if (MF_SIMILARITY_ITEM) {
		unsigned count = 0;
		for (unsigned int u = 0; u < items.size(); ++u) {
			for (unsigned v = u + 1; v < items.size(); ++v) {

				double value = 0;

				switch (MF_SIMILARITY_ITEM) {
				case LIANG_SIMILARITY:
					value = get_liang_item_similarity(u, v);
					break;
				default:
					value = calc_similarity_item(u, v, MF_SIMILARITY_ITEM);
					break;
				}

				if (abs(value) > threshold) {
					pair_similarity *pair = new pair_similarity();
					pair->u = u;
					pair->v = v;
					pair->s = value;
					//pair.weight = calc_similarity_item(u, v, INTERSECTION) + 1;
					pair->weight = 1;

					neighborhood_item[u].push_back(pair);
					neighborhood_item[v].push_back(pair);

					++count;
				}
			}
		}
	}

	for (unsigned int it = 0; it < MF_NUM_ITERATIONS; ++it) {

		for (unsigned int i = 0; i < users.size(); ++i) {
			User *u = users[i];

			for (unsigned k = 0; k < MF_NUM_FACTORS; ++k) {

				double numerator = 0;
				double denominator = 0;

				for (unsigned int b = 0; b < u->ratings.size(); ++b) {
					Vote *v = u->ratings[b];
					unsigned int j = v->itemId;
					double rij = v->rating;

					double eij = (rij
							- (dot_product(p[i], q[j]) - p[i][k] * q[j][k]));
					numerator += q[j][k] * eij;
					denominator += pow(q[j][k], 2);
				}

				double sim_sum = 0;

				double aux = 0;
				if (MF_SIMILARITY_USER) {

					for (unsigned b = 0; b < neighborhood_user[u->id].size();
							++b) {
						pair_similarity *p_sim = neighborhood_user[u->id][b];

						unsigned target =
								u->id == p_sim->u ? p_sim->v : p_sim->u;
						sim_sum += p_sim->s;

						numerator += p_sim->s * p[target][k];

						aux += p_sim->s * p[target][k];
					}
				}
				cout << aux << ",";
				p[i][k] = numerator / (MF_LAMBDA + denominator + sim_sum);

			}
			cout << endl;

		}

		for (unsigned int j = 0; j < items.size(); ++j) {
			for (unsigned int k = 0; k < MF_NUM_FACTORS; ++k) {

				double numerator = 0;
				double denominator = 0;

				for (unsigned int b = 0; b < items[j]->ratings.size(); ++b) {
					Vote *v = items[j]->ratings[b];
					User *u = users[v->userId];
					unsigned int i = u->id;

					double rij = v->rating;

					double eij = (rij
							- (dot_product(p[i], q[j]) - p[i][k] * q[j][k]));
					numerator += p[i][k] * eij;
					denominator += pow(p[i][k], 2);
				}

				double sim_sum = 0;
				if (MF_SIMILARITY_ITEM) {

					for (unsigned b = 0; b < neighborhood_item[j].size(); ++b) {
						pair_similarity *p_sim = neighborhood_item[j][b];

						unsigned target = j == p_sim->u ? p_sim->v : p_sim->u;
						sim_sum += p_sim->s;

						numerator += p_sim->s * q[target][k];
					}
				}

				q[j][k] = numerator / (MF_LAMBDA + denominator + sim_sum);
			}
		}
	}
}

void sgd_smf_new_model(const vector<Vote *> &trainingset,
		vector<vector<double> > &p, vector<vector<double> > &q) {

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

// calcula a similaridade dos usuarios
	if (MF_SIMILARITY_USER) {

		int count = 0;
		for (unsigned int u = 0; u < users.size(); ++u) {
			for (unsigned v = u + 1; v < users.size(); ++v) {
				double value = 0;

				switch (MF_SIMILARITY_USER) {
				case LIANG_SIMILARITY:
					value = get_liang_user_similarity(u, v);
					break;
				default:
					value = calc_similarity_user(u, v, MF_SIMILARITY_USER);
					break;
				}

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

//calcula a similaridade dos itens
	if (MF_SIMILARITY_ITEM) {
		int count = 0;
		for (unsigned int u = 0; u < items.size(); ++u) {
			for (unsigned v = u + 1; v < items.size(); ++v) {

				double value = 0;

				switch (MF_SIMILARITY_ITEM) {
				case LIANG_SIMILARITY:
					value = get_liang_item_similarity(u, v);
					break;
				default:
					value = calc_similarity_item(u, v, MF_SIMILARITY_ITEM);
					break;
				}

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

	vector<vector<double> > p_gradient(users.size(),
			vector<double>(MF_NUM_FACTORS, 0.0));
	vector<vector<double> > q_gradient(items.size(),
			vector<double>(MF_NUM_FACTORS, 0.0));

	for (unsigned int it = 0; it < MF_NUM_ITERATIONS; ++it) {
		random_shuffle(training_indexes.begin(), training_indexes.end());

#pragma omp parallel num_threads(NUM_THREADS)
		{

#pragma omp for

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
					double temp = p[u_id][k];
//					p[u_id][k] += MF_ALPHA
//							* (err * q[i_id][k] - MF_LAMBDA * p[u_id][k]);
//					q[i_id][k] += MF_ALPHA
//							* (err * temp - MF_LAMBDA * q[i_id][k]);

					p_gradient[u_id][k] += -err * q[i_id][k];
					q_gradient[i_id][k] += -err * p[u_id][k];
				}
			}
		}

		if (MF_SIMILARITY_USER) {
			random_shuffle(pair_indexes_user.begin(), pair_indexes_user.end());

#pragma omp parallel num_threads(NUM_THREADS)
			{

#pragma omp for
				for (unsigned int i = 0; i < pair_indexes_user.size(); ++i) {

					unsigned int idx = pair_indexes_user[i];

					const unsigned int u = neighborhood_user[idx].u;
					const unsigned int v = neighborhood_user[idx].v;
					const double s = neighborhood_user[idx].s;
					const double w = neighborhood_user[idx].weight;

					for (unsigned int k = 0; k < MF_NUM_FACTORS; ++k) {
//						double step = MF_ALPHA * s * (p[u][k] - p[v][k]);
//						p[u][k] -= step;
//						p[v][k] -= -step;

						double err = s * (p[u][k] - p[v][k]);
						p_gradient[u][k] += err;
						p_gradient[v][k] += -err;
					}
				}
			}
		}

		if (MF_SIMILARITY_ITEM) {
			random_shuffle(pair_indexes_item.begin(), pair_indexes_item.end());

#pragma omp parallel num_threads(NUM_THREADS)
			{

#pragma omp for
				for (unsigned int i = 0; i < pair_indexes_item.size(); ++i) {

					unsigned int idx = pair_indexes_item[i];
					const unsigned int u = neighborhood_item[idx].u;
					const unsigned int v = neighborhood_item[idx].v;
					const double s = neighborhood_item[idx].s;
					const double w = neighborhood_item[idx].weight;

					for (unsigned int k = 0; k < MF_NUM_FACTORS; ++k) {
//						double step = MF_ALPHA * s * (q[u][k] - q[v][k]);
//						q[u][k] -= step;
//						q[v][k] -= -step;
						double err = s * (q[u][k] - q[v][k]);
						q_gradient[u][k] += err;
						q_gradient[v][k] += -err;
					}
				}
			}
		}

		{
			for (unsigned int i = 0; i < users.size(); ++i) {
				for (unsigned int k = 0; k < MF_NUM_FACTORS; ++k) {
					p[i][k] -= MF_ALPHA
							* (p_gradient[i][k] + MF_LAMBDA * p[i][k]);

					p_gradient[i][k] = 0;
				}
			}

			for (unsigned int i = 0; i < items.size(); ++i) {
				for (unsigned int k = 0; k < MF_NUM_FACTORS; ++k) {
					q[i][k] -= MF_ALPHA
							* (q_gradient[i][k] + MF_LAMBDA * q[i][k]);
					q_gradient[i][k] = 0;
				}
			}
		}
	}
}

void sgd_smf_asymmetric(const vector<Vote *> &trainingset,
		vector<vector<double> > &p, vector<vector<double> > &q,
		vector<vector<double> > &y, vector<vector<double> > &z) {

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
				{
					// (u,v)
					pair_similarity pair;
					pair.u = u;
					pair.v = v;
					pair.s = value;
					pair.weight = 1;

					switch (MF_SIMILARITY_USER) {
					case ASYMMETRIC_JACCARD_SIMILARITY:
						pair.s /= users[u]->ratings.size();
						break;

					case ASYMMETRIC_ADAMIC_ADAR_SIMILARIY:
						pair.s /= users[u]->adamic_adar_sum_neighbors;
						break;
					}

					if (abs(pair.s) > threshold) {
						neighborhood_user.push_back(pair);
						pair_indexes_user.push_back(count++);
					}
				}

				{
					// (v,u)
					pair_similarity pair;
					pair.u = v;
					pair.v = u;
					pair.s = value;
					pair.weight = 1;

					switch (MF_SIMILARITY_USER) {
					case ASYMMETRIC_JACCARD_SIMILARITY:
						pair.s /= users[v]->ratings.size();
						break;

					case ASYMMETRIC_ADAMIC_ADAR_SIMILARIY:
						pair.s /= users[v]->adamic_adar_sum_neighbors;
						break;
					case ACOS:
						pair.s = calc_similarity_user(v, u, MF_SIMILARITY_USER);
						break;
					case AMSD:
						pair.s = calc_similarity_user(v, u, MF_SIMILARITY_USER);
						break;
					}

					if (abs(pair.s) > threshold) {

						neighborhood_user.push_back(pair);
						pair_indexes_user.push_back(count++);
					}
				}
			}
		}
	}

	if (MF_SIMILARITY_ITEM) {
		int count = 0;
		for (unsigned int u = 0; u < items.size(); ++u) {
			for (unsigned v = u + 1; v < items.size(); ++v) {

				double value = calc_similarity_item(u, v, MF_SIMILARITY_ITEM);

				{
					// (u,v)
					pair_similarity pair;
					pair.u = u;
					pair.v = v;
					pair.s = value;
					pair.weight = 1;

					switch (MF_SIMILARITY_ITEM) {
					case ASYMMETRIC_JACCARD_SIMILARITY:
						pair.s /= items[u]->ratings.size();
						break;

					case ASYMMETRIC_ADAMIC_ADAR_SIMILARIY:
						pair.s /= items[u]->adamic_adar_sum_neighbors;
						break;
					}
					if (abs(pair.s) > threshold) {
						neighborhood_item.push_back(pair);
						pair_indexes_item.push_back(count++);

					}
				}
				{
					// (v,u)
					pair_similarity pair;
					pair.u = v;
					pair.v = u;
					pair.s = value;
					//pair.weight = calc_similarity_item(u, v, INTERSECTION) + 1;
					pair.weight = 1;

					switch (MF_SIMILARITY_ITEM) {
					case ASYMMETRIC_JACCARD_SIMILARITY:
						pair.s /= items[v]->ratings.size();
						break;

					case ASYMMETRIC_ADAMIC_ADAR_SIMILARIY:
						pair.s /= items[v]->adamic_adar_sum_neighbors;
						break;
					case ACOS:
						pair.s = calc_similarity_item(v, u, MF_SIMILARITY_ITEM);
						break;
					case AMSD:
						pair.s = calc_similarity_item(v, u, MF_SIMILARITY_ITEM);
						break;
					}
					if (abs(pair.s) > threshold) {

						neighborhood_item.push_back(pair);
						pair_indexes_item.push_back(count++);
					}
				}
			}
		}
	}

	vector<vector<double> > p_gradient(users.size(),
			vector<double>(MF_NUM_FACTORS, 0.0));
	vector<vector<double> > y_gradient(users.size(),
			vector<double>(MF_NUM_FACTORS, 0.0));

	vector<vector<double> > q_gradient(items.size(),
			vector<double>(MF_NUM_FACTORS, 0.0));
	vector<vector<double> > z_gradient(items.size(),
			vector<double>(MF_NUM_FACTORS, 0.0));

	double previous = DBL_MAX;
	for (unsigned int it = 0; it < MF_NUM_ITERATIONS; ++it) {

		random_shuffle(training_indexes.begin(), training_indexes.end());

#pragma omp parallel num_threads(NUM_THREADS)
		{

#pragma omp for

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
//					double temp = p[u_id][k];
//					p[u_id][k] += MF_ALPHA
//							* (err * q[i_id][k] - MF_LAMBDA * p[u_id][k]);
//					q[i_id][k] += MF_ALPHA
//							* (err * temp - MF_LAMBDA * q[i_id][k]);

					p_gradient[u_id][k] += -err * q[i_id][k];
					q_gradient[i_id][k] += -err * p[u_id][k];
				}
			}
		}

//TODO: asymmetric functionality must be implemented here
//A fatoração precisa. No paper que lhe passei eu fiz suv = pu*PV. Precisamos fazer SUV=PU.tv

		if (MF_SIMILARITY_USER) {

			random_shuffle(pair_indexes_user.begin(), pair_indexes_user.end());

#pragma omp parallel num_threads(NUM_THREADS)
			{

#pragma omp for
				for (unsigned int i = 0; i < pair_indexes_user.size(); ++i) {

					unsigned int idx = pair_indexes_user[i];

					const unsigned int u = neighborhood_user[idx].u;
					const unsigned int v = neighborhood_user[idx].v;
					const double s = neighborhood_user[idx].s;
					const double w = neighborhood_user[idx].weight;

					double sig_p = dot_product(p[u], y[v]);
					double err = (s - sig_p);
					if (MF_NORMALIZE) {
						sig_p = sigmoid(dot_product(p[u], y[v]));
						err = (s - sig_p) * sig_p * (1 - sig_p);
					}

					for (unsigned int k = 0; k < MF_NUM_FACTORS; ++k) {
						double temp = p[u][k];
//						p[u][k] += MF_ALPHA * (w * err * y[v][k]);
//						y[v][k] += MF_ALPHA
//								* (w * err * temp - MF_LAMBDA * y[v][k]);

						p_gradient[u][k] += -err * y[v][k];
						y_gradient[v][k] += -err * p[u][k];

					}
				}
			}
		}

		if (MF_SIMILARITY_ITEM) {
			random_shuffle(pair_indexes_item.begin(), pair_indexes_item.end());

#pragma omp parallel num_threads(NUM_THREADS)
			{

#pragma omp for
				for (unsigned int i = 0; i < pair_indexes_item.size(); ++i) {

					unsigned int idx = pair_indexes_item[i];
					const unsigned int u = neighborhood_item[idx].u;
					const unsigned int v = neighborhood_item[idx].v;
					const double s = neighborhood_item[idx].s;
					const double w = neighborhood_item[idx].weight;

					double sig_p = dot_product(q[u], z[v]);
					double err = (s - sig_p);
					if (MF_NORMALIZE) {
						sig_p = sigmoid(dot_product(q[u], z[v]));
						err = (s - sig_p) * sig_p * (1 - sig_p);
					}

					for (unsigned int k = 0; k < MF_NUM_FACTORS; ++k) {
//						double temp = q[u][k];
//						q[u][k] += MF_ALPHA * (w * err * z[v][k]);
//						z[v][k] += MF_ALPHA
//								* (w * err * temp - MF_LAMBDA * z[v][k]);

						q_gradient[u][k] += -err * z[v][k];
						z_gradient[v][k] += -err * q[u][k];
					}
				}
			}
		}

		if (true) {
			for (unsigned int i = 0; i < users.size(); ++i) {
				for (unsigned int k = 0; k < MF_NUM_FACTORS; ++k) {
					p[i][k] -= MF_ALPHA
							* (p_gradient[i][k] + MF_LAMBDA * p[i][k]);
					y[i][k] -= MF_ALPHA
							* (y_gradient[i][k] + MF_LAMBDA * y[i][k]);

					p_gradient[i][k] = 0;
					y_gradient[i][k] = 0;
				}
			}

			for (unsigned int i = 0; i < items.size(); ++i) {
				for (unsigned int k = 0; k < MF_NUM_FACTORS; ++k) {
					q[i][k] -= MF_ALPHA
							* (q_gradient[i][k] + MF_LAMBDA * q[i][k]);
					z[i][k] -= MF_ALPHA
							* (z_gradient[i][k] + MF_LAMBDA * z[i][k]);

					q_gradient[i][k] = 0;
					z_gradient[i][k] = 0;
				}
			}
		}


		if (false) {
			double global_error = 0;
			for (unsigned int i = 0; i < SIZE; ++i) {
				Vote *v = trainingset[training_indexes[i]];

				unsigned int u_id = v->userId;
				unsigned int i_id = v->itemId;

				double r = v->rating;
				double err = (r - dot_product(p[u_id], q[i_id]));

				global_error += pow(err, 2);
			}

			global_error = sqrt(global_error / SIZE);
			cout << global_error << endl;
			double diff = global_error - previous;
//			if (abs(diff) < 0.001 || diff > 0) {
//				return;
//			}
			previous = global_error;
		}
	}
}

void read_social(const char *filename) {
	ifstream file(filename);
	string line;

	unsigned int userId;
	unsigned int itemId;
	float rating;

	getline(file, line); // reading header

	while (getline(file, line)) {
		stringstream ss(line);
		string tok;

		getline(ss, tok, DELIM);
		userId = atoi(tok.c_str());

		getline(ss, tok, DELIM);
		itemId = atoi(tok.c_str());

		if (social_network.find(userId) == social_network.end()) {
			social_network[userId] = vector<unsigned>();

		}
		social_network[userId].push_back(itemId);
	}
}

void run_matrix_factorization(vector<TestItem> &test,
		vector<Vote *> &trainingset, const unsigned int fold) {

	std::normal_distribution<double> distribution(0, 0.1);

	vector<vector<double> > p(users.size(),
			vector<double>(MF_NUM_FACTORS, 0.0));
	vector<vector<double> > y(users.size(),
			vector<double>(MF_NUM_FACTORS, 0.0));

	for (unsigned int i = 0; i < users.size(); ++i) {
		for (unsigned int k = 0; k < MF_NUM_FACTORS; ++k) {
			p[i][k] = distribution(generator);
			y[i][k] = distribution(generator);
		}
	}

	vector<vector<double> > q(items.size(),
			vector<double>(MF_NUM_FACTORS, 0.0));
	vector<vector<double> > z(items.size(),
			vector<double>(MF_NUM_FACTORS, 0.0));

	for (unsigned int i = 0; i < items.size(); ++i) {
		for (unsigned int k = 0; k < MF_NUM_FACTORS; ++k) {
			q[i][k] = distribution(generator);
			z[i][k] = distribution(generator);
		}
	}

	string output_file;
	switch (MF_ALGORITHM) {
	case 3: {
		als_smf_new_model(trainingset, p, q);
		break;
	}
	case 4: {
		sgd_smf(trainingset, p, q);
		break;
	}
	case 5: {
		sgd_smf_asymmetric(trainingset, p, q, y, z);
		break;
	}
	case 6: {
		read_social("sn.csv");

		map<unsigned int, vector<unsigned> > user_neighbors;

		// itera por cada usuario do treino e monta lista de vizinhanca de usuarios do treino
		for (unsigned int u = 0; u < users.size(); ++u) {
			if (rUserIds.find(u) != rUserIds.end()) {
				unsigned idx = rUserIds[u];
				vector<unsigned> u_neigh;

				if (social_network.find(idx) != social_network.end()) {
					for (unsigned j = 0; j < social_network[idx].size(); ++j) {
						unsigned idx2 = social_network[idx][j];
						if (userIds.find(idx2) != userIds.end()) {
							u_neigh.push_back(userIds[idx2]);
						}
					}
				}

				if (u_neigh.size() > 0) {
					user_neighbors[u] = u_neigh;
				}
			}
		}

		sgd_social_MF(trainingset, user_neighbors, p, q);
		break;
	}
	case 7: {
		smf(trainingset, p, q);
		break;
	}
	case 8: {
		read_social("sn.csv");

		map<unsigned int, vector<unsigned> > user_neighbors;

		// itera por cada usuario do treino e monta lista de vizinhanca de usuarios do treino
		for (unsigned int u = 0; u < users.size(); ++u) {
			if (rUserIds.find(u) != rUserIds.end()) {
				unsigned idx = rUserIds[u];
				vector<unsigned> u_neigh;

				if (social_network.find(idx) != social_network.end()) {
					for (unsigned j = 0; j < social_network[idx].size(); ++j) {
						unsigned idx2 = social_network[idx][j];
						if (userIds.find(idx2) != userIds.end()) {
							u_neigh.push_back(userIds[idx2]);
						}
					}
				}

				if (u_neigh.size() > 0) {
					user_neighbors[u] = u_neigh;
				}
			}
		}

		sgd_smf_social_similarity(trainingset, p, q, user_neighbors);
		break;
	}
	}

	char buffer[250];
	sprintf(buffer, "%u-%u-%u-%u-%3.5f-%3.5f", MF_NUM_FACTORS,
			MF_SIMILARITY_USER, MF_SIMILARITY_ITEM, MF_NORMALIZE, MF_LAMBDA,
			MF_ALPHA);

	output_file = string(buffer);

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

	MAX_RATING = -INT_MAX;
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
void scaleUser(User *u) {
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
			scaleUser(users[i]);
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

		switch (MF_SIMILARITY_ITEM) {

		case LIANG_SIMILARITY:
			unsigned size = items.size();
			item_similarity_matrix.resize(0.5 * (size - 1) * size, -2);
			calc_liang_item_similarity();
			break;
		}

		switch (MF_SIMILARITY_USER) {

		case LIANG_SIMILARITY:
			unsigned size = users.size();
			user_similarity_matrix.resize(0.5 * (size - 1) * size, -2);
			calc_liang_user_similarity();
			break;
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
	case 0: {
		generate_dataset();
		break;
	}
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
