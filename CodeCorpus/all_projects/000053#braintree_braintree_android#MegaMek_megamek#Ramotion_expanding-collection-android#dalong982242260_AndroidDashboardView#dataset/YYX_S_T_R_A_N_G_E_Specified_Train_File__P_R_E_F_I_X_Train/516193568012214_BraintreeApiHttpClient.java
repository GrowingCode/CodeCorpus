package com.braintreepayments.api.internal;

import android.text.TextUtils;

import com.braintreepayments.api.BuildConfig;
import com.braintreepayments.api.exceptions.BraintreeApiErrorResponse;
import com.braintreepayments.api.exceptions.UnprocessableEntityException;
import com.braintreepayments.api.interfaces.HttpResponseCallback;

import java.io.IOException;
import java.net.HttpURLConnection;

import javax.net.ssl.SSLException;

/**
 * Network request class that handles BraintreeApi request specifics and threading.
 */
public class BraintreeApiHttpClient extends HttpClient {

    public static final String API_VERSION_2016_10_07 = "2016-10-07";

    private final String mAccessToken;
    private final String mApiVersion;

    public BraintreeApiHttpClient(String baseUrl, String accessToken) {
        super();

        mBaseUrl = baseUrl;
        mAccessToken = accessToken;
        mApiVersion = API_VERSION_2016_10_07;

        setUserAgent("braintree/android/" + BuildConfig.VERSION_NAME);

        try {
            setSSLSocketFactory(new TLSSocketFactory(BraintreeApiCertificate.getCertInputStream()));
        } catch (SSLException e) {
            setSSLSocketFactory(null);
        }
    }

    @Override
    protected HttpURLConnection init(String url) throws IOException {
        HttpURLConnection connection = super.init(url);

        if (!TextUtils.isEmpty(mAccessToken)) {
            connection.setRequestProperty("Authorization", "Bearer " + mAccessToken);
        }

        connection.setRequestProperty("Braintree-Version", mApiVersion);

        return connection;
    }

    @Override
    protected String parseResponse(HttpURLConnection connection) throws Exception {
        try {
            return super.parseResponse(connection);
        } catch (UnprocessableEntityException e) {
            throw new BraintreeApiErrorResponse(e.getMessage());
        }
    }
}
