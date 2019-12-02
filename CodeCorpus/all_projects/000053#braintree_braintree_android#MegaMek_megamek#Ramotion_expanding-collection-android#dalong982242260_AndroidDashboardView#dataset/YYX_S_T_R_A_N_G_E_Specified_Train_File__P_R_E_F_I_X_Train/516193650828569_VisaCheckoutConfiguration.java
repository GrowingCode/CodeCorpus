package com.braintreepayments.api.models;

import com.braintreepayments.api.Json;
import com.visa.checkout.Profile.CardBrand;
import com.visa.checkout.VisaCheckoutSdk;

import org.json.JSONObject;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;

/**
 * Contains the remote Visa Checkout configuration for the Braintree SDK.
 */
public class VisaCheckoutConfiguration {

    private boolean mIsEnabled;
    private String mApiKey;
    private String mExternalClientId;
    private List<String> mCardBrands;

    /**
     * Determines if the Visa Checkout SDK is available.
     * @return true when the class can be found, false otherwise.
     */
    private static boolean isVisaCheckoutSDKAvailable() {
        try {
            Class.forName(VisaCheckoutSdk.class.getName());
            return true;
        } catch (ClassNotFoundException | NoClassDefFoundError e) {
            return false;
        }
    }

    static VisaCheckoutConfiguration fromJson(JSONObject json) {
        VisaCheckoutConfiguration visaCheckoutConfiguration = new VisaCheckoutConfiguration();

        if (json == null) {
            json = new JSONObject();
        }

        visaCheckoutConfiguration.mApiKey = Json.optString(json, "apikey", "");
        visaCheckoutConfiguration.mIsEnabled = isVisaCheckoutSDKAvailable() && visaCheckoutConfiguration.mApiKey != "";
        visaCheckoutConfiguration.mExternalClientId = Json.optString(json, "externalClientId", "");
        visaCheckoutConfiguration.mCardBrands = supportedCardTypesToAcceptedCardBrands(
                CardConfiguration.fromJson(json).getSupportedCardTypes());

        return visaCheckoutConfiguration;
    }

    /**
     * Determines if the Visa Checkout flow is available to be used. This can be used to determine
     * if UI components should be shown or hidden.
     *
     * @return boolean if Visa Checkout SDK is available, and configuration is enabled.
     */
    public boolean isEnabled() {
        return mIsEnabled;
    }

    /**
     * @return The Visa Checkout External Client Id associated with this merchant's Visa Checkout configuration.
     */
    public String getExternalClientId() {
        return mExternalClientId;
    }

    /**
     * @return The Visa Checkout API Key associated with this merchant's Visa Checkout configuration.
     */
    public String getApiKey() {
        return mApiKey;
    }

    /**
     * @return The accepted card brands for Visa Checkout.
     */
    public List<String> getAcceptedCardBrands() {
        return mCardBrands;
    }

    private static List<String> supportedCardTypesToAcceptedCardBrands(
            Set<String> supportedCardTypes) {
        List<String> acceptedCardBrands = new ArrayList<>();
        for (String supportedCardType : supportedCardTypes) {
            switch (supportedCardType.toLowerCase()) {
                case "visa":
                    acceptedCardBrands.add(CardBrand.VISA);
                    break;
                case "mastercard":
                    acceptedCardBrands.add(CardBrand.MASTERCARD);
                    break;
                case "discover":
                    acceptedCardBrands.add(CardBrand.DISCOVER);
                    break;
                case "american express":
                    acceptedCardBrands.add(CardBrand.AMEX);
                    break;
            }
        }
        return acceptedCardBrands;
    }
}
